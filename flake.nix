{
  description = "ReTrack - CPU-efficient multi-object tracking with selective ReID";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python312
            uv

            # Native build tooling
            gcc
            pkg-config
            git
          ];

          shellHook = ''
            export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
          '';
        };
      }
    );
}
