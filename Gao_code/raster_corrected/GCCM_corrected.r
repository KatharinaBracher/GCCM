GCCMSingle<-function(xEmbedings,yPred,lib_size,pred,totalRow,totalCol,b) {
  # yPred: yMatrix
  # pred: pixel indices selected for prediction

  x_xmap_y <- data.frame()
  
  # sliding window (library) over the data. 
  for(r in 1:(totalRow-lib_size+1))
  {
    for(c in 1:(totalCol-lib_size+1))
    {
      
      # initialize boolean vector, indicates for each pixel
      pred_indices <- rep.int(FALSE, times = totalRow*totalCol) 
      lib_indices<- rep.int(FALSE, times = totalRow*totalCol)
      
      pred_indices[locate(pred[,1],pred[,2],totalRow,totalCol)]<-TRUE # indicating which pixels in the total matrix are for prediction
      
      pred_indices[which(is.na(yPred)) ]<-FALSE # Ensures prediction indices do not include NA values in yPred.
      
      
      lib_rows<-seq(r,(r+lib_size-1))
      lib_cols<-seq(c,(c+lib_size-1))
      
      lib_ids<-merge(lib_rows,lib_cols) # window of indices considered in current library

      
      lib_indices[locate(lib_ids[,1],lib_ids[,2],totalRow,totalCol)]<-TRUE  # indicating which pixels in current library

      # Skips to the next iteration if more than half of the values in the library indices are NA
      if(length(which(is.na(yPred[which(lib_indices)]))) > ((lib_size*lib_size)/2))
      {
        next
      }
      
      # run cross map and store results
      results <-  projection(xEmbedings,yPred,lib_indices,pred_indices,b)
      
      
      x_xmap_y <- rbind(x_xmap_y, data.frame(L = lib_size, rho = results$stats$rho)) 
      
    }
    
  }
  
  return(x_xmap_y)
}


GCCM<-function(xMatrix, yMatrix, lib_sizes, lib, pred, E, tau = 1, b = E+1,cores=NULL)
{
  imageSize<-dim(xMatrix)
  totalRow<-imageSize[1]
  totalCol<-imageSize[2]
  
  yPred<- as.vector(t(yMatrix)) #####!!!!####### this should be as vector not as.array to flatten (row major bc of T)
  
  xEmbedings<-list()
  xEmbedings[[1]]<- as.vector(t(xMatrix)) ######!!!!####### NOW FLAT this should be as vector not as matrix to flatten (row major bc of T)
  
  ########################################!!!!!!!!!!!!######################################
  # this should be E-1 because the first dimension is the focal unit
  # the original code results in a E+1 dimensional embedding
  for(i in 1:(E-1)) {
    xEmbedings[[i+1]]<-laggedVariableAs2Dim(xMatrix, i)  #### row first
  }
  
  x_xmap_y <- data.frame()
  
  if(is.null(cores))
  {
    for(lib_size in lib_sizes)
    {
      x_xmap_y<-rbind(x_xmap_y,GCCMSingle(xEmbedings,yPred,lib_size,pred,totalRow,totalCol,b))
      
    }
  }else
  {
    cl <- makeCluster(cores)
    registerDoParallel(cl)
    clusterExport(cl,deparse(substitute(GCCMSingle)))
    clusterExport(cl,deparse(substitute(locate)))
    clusterExport(cl,deparse(substitute(projection)))
    clusterExport(cl,deparse(substitute(distance_Com)))
    clusterExport(cl,deparse(substitute(compute_stats)))
    
    x_xmap_y <- foreach(lib_size=lib_sizes, .combine='rbind') %dopar% GCCMSingle(xEmbedings,yPred,lib_size,pred,totalRow,totalCol,b)
    stopCluster(cl)
  }
  
  
  return (x_xmap_y)
}

# equivalent to ravel function in python 
locate<-function(curRow,curCOl,totalRow,totalCol) # Converts a matrix-style row and column index into a linear index
{ # this is converting to a linear index in row major style
  return ((curRow-1)*totalCol+curCOl) 
}


projection<-function(embedings,target,lib_indices, pred_indices,num_neighbors)
{
  pred <- rep.int(NaN, times = length(target))
  
  for(p in which(pred_indices)) # iterating over all pixel indices that are to be predicted (TRUE)
  {
    # Temporarily removes the prediction point from the library to ensure the model does not use its own value in prediction
    temp_lib <- lib_indices[p]
    lib_indices[p] <- FALSE
    
    libs <- which(lib_indices)
    
    # compute distances between the embedding of the prediction point and embeddings of all points in the adjusted library.
    distances<-distance_Com(embedings,libs,p)
    
    # find nearest neighbors
    neighbors <- order(distances)[1:num_neighbors]
    min_distance <- distances[neighbors[1]]
    if(is.na(min_distance))
    {
      
      lib_indices[p] <- temp_lib 
      next
    }
    # compute weights
    if(min_distance == 0) # perfect match
    {
      weights <- rep.int(0.000001, times = num_neighbors)
      weights[distances[neighbors] == 0] <- 1
    }
    else
    {
      weights <- exp(-distances[neighbors]/min_distance)
      weights[weights < 0.000001] <- 0.000001
    }
    total_weight <- sum(weights)
    
    # make prediction
    # weighted average of the target values at the neighbor locations, using the calculated weights
    ############# correct now bc of flatten yPred row major
    pred[p] <- (weights %*% target[libs[neighbors]]) / total_weight
    
    
    
    lib_indices[p] <- temp_lib 
  }
  ############# correct now bc of flatten yPred row major
  return(list(pred = pred, stats = compute_stats(target[pred_indices], pred[pred_indices])))
  
}


distance_Com<-function(embeddings,libs,p)
{
  distances<-c()
  
  
  #calculate distances of first dimension
  emd <- embeddings[[1]] ################## now correct now bc of flatten xMatrix row major
  distances<-cbind(distances,abs(emd[libs]-emd[p]))

  for(e in 2:length(embeddings))
  {
    emd<-embeddings[[e]]
    # ????????? this is used nowhere
    q <- matrix(rep(emd[p], length(libs)), nrow = length(libs), byrow = T)
    ###################################################### added 
    dist <- sweep(emd[libs,], 2, emd[p,], FUN = "-") # element wise distance
    mean_dist <- rowMeans(dist,na.rm=TRUE) # mean over neighbors
    distances<-cbind(distances,abs(mean_dist))
  }
  
  return (rowMeans(distances,na.rm=TRUE)) # mean over all embedding orders to get one distance for each unit to focal unit
  
}

