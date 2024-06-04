{ pkgs ? import <nixpkgs> { } }:
with pkgs;
let
  pkgs21 = import
    (builtins.fetchTarball {
      url = "https://github.com/NixOS/nixpkgs/archive/4d2b37a84fad1091b9de401eb450aae66f1a741e.tar.gz";
    })
    { };
in
pkgs.mkShell {
  nativeBuildInputs = [
    pkgs21.R
    pkgs21.rPackages.rgdal
    pkgs21.rPackages.foreach
    pkgs21.rPackages.doParallel
  ];
  shellHook = ''
  '';
}
