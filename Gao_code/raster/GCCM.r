GCCMSingle<-function(xEmbedings,yPred,lib_size,pred,totalRow,totalCol,b) {
  # yPred: yMatrix
  # pred: pixel indices selected for prediction

  x_xmap_y <- data.frame()
  
  # sliding window over the data. 
  # The bounds ensure that the library (a subset of the total data) fits within the data matrix dimensions.
  for(r in 1:(totalRow-lib_size+1))
  {
    for(c in 1:(totalCol-lib_size+1))
    {
      
      # initialize boolean vector, indicates for each pixel
      pred_indices <- rep.int(FALSE, times = totalRow*totalCol) 
      lib_indices<- rep.int(FALSE, times = totalRow*totalCol)
      
      pred_indices[locate(pred[,1],pred[,2],totalRow,totalCol) ]<-TRUE # indicating which pixels in the total matrix are for prediction
      
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
  
  yPred<- as.array(t(yMatrix))
  
  xEmbedings<-list()
  xEmbedings[[1]]<- as.array(t(xMatrix)) # focal units s
  
  for(i in 1:E) {
    # s(1), s(2), ..., s(E)
    # matrix where row corresponds to each pixel and columns are neighbors of order i 
    xEmbedings[[i+1]]<-laggedVariableAs2Dim(xMatrix, i)  #### row first
    # [1] col x row focal
    # [2] col*row x 8 1st order
    # [3] col*row x 16 2nd order
  
  }
  
  x_xmap_y <- data.frame()
  
  if(is.null(cores))
  {
    for(lib_size in lib_sizes)
    {
      # calculate rho for each pred pixel: returns L          rho
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


locate<-function(curRow,curCOl,totalRow,totalCol) # Converts a matrix-style row and column index into a linear index
{
  return ((curRow-1)*totalCol+curCOl) 
}


projection<-function(embedings,target,lib_indices, pred_indices,num_neighbors)
{
  #Exclusion of Current Prediction Point: Temporarily removes the current prediction point from the library to prevent leakage.
  #Distance Calculation: Uses the distance_Com function to calculate distances between the embedding of the current prediction point and all library points.
  #Neighbor Selection and Weight Calculation: Identifies the nearest neighbors based on the distances, calculates weights for these neighbors (inversely proportional to the distance), and handles cases of perfect matches.
  #Prediction: Computes a weighted average of the target values at the neighbor points to predict the target at the current prediction point.
  #Statistics: Calculates statistics for the predictions, likely evaluating the prediction accuracy or fit.
  
  pred <- rep.int(NaN, times = length(target))
  
  for(p in which(pred_indices)) # iterating over all pixel indices that are to be predicted (TRUE)
  {
    # Temporarily removes the prediction point from the library to ensure the model does not use its own value in prediction
    temp_lib <- lib_indices[p]
    lib_indices[p] <- FALSE
    
    libs <- which(lib_indices)
    
    # compute distances between the embedding of the prediction point and embeddings of all points in the adjusted library.
    distances<-distance_Com(embedings,libs,p)
    
    #distances<-colMeans(distances)
    
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
    pred[p] <- (weights %*% target[libs[neighbors]]) / total_weight
    
    
    
    lib_indices[p] <- temp_lib 
  }
  
  # return output & stats
  return(list(pred = pred, stats = compute_stats(target[pred_indices], pred[pred_indices])))
  
}


distance_Com<-function(embeddings,libs,p)
{
  distances<-c()

  for(e in 1:length(embeddings))
  {
    emd<-embeddings[[e]]
    
    q <- matrix(rep(emd[p], length(libs)), nrow = length(libs), byrow = T)
  
    
    distances<-cbind(distances,abs(emd[libs]-emd[p]))
    
  }
  
  return (rowMeans(distances,na.rm=TRUE))
  
}

