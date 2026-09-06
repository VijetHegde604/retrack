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

          # Dynamically link the libraries uv/pip wheels expect
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
            stdenv.cc.cc.lib # Provides libstdc++.so.6 for numpy/cv2
            zlib             # Core compression library
            glib             # Required by OpenCV
            libGL            # Required by OpenCV
          ]);

          shellHook = ''
            export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
          '';
        };
      }
    );
}
