{ pkgs ? import <nixpkgs> { } }:
with pkgs;
let
  pkgs21 = import
    (builtins.fetchTarball {
      url = "https://github.com/NixOS/nixpkgs/archive/4d2b37a84fad1091b9de401eb450aae66f1a741e.tar.gz";
    })
    { };
in
pkgs.mkShell rec {
  buildInputs = [
    pkgs.python3
    pkgs.poetry
    pkgs.zlib
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
  '';
}