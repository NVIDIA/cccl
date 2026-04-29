{
  description = "Static analysis infrastructure for CCCL";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        analysisTargets = import ./nix/analysis {
          inherit pkgs;
          inherit (pkgs) lib;
          src = self;
        };
      in
      {
        packages = analysisTargets;

        devShells.default = import ./nix/dev-shell.nix {
          inherit pkgs;
        };
      }
    );
}
