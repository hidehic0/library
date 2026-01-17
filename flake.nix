{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        cpp-dump = pkgs.callPackage ./nix/cpp-dump/default.nix { };
      in
      {
        devShells.default = pkgs.mkShell {
          packages =
            with pkgs;
            [
              uv
              ac-library
            ]
            ++ [
              cpp-dump
            ];
          shellHook = ''
            source .venv/bin/activate
            export PYTHONPATH=$(pwd)
          '';
        };
      }
    );
}
