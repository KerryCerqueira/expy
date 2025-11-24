{
  description = "Reproducible AI experiments with python";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };
  outputs = {
    self,
    nixpkgs,
    flake-parts,
    ...
  }@inputs: let
    pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
    buildExpyFrom = ps:
      ps.python3Packages.buildPythonPackage {
        pname = "expy";
        inherit (pyproject.project) version;
        format = "pyproject";
        src = ./.;
        build-system = with ps.python3Packages; [
          setuptools
        ];
        propagatedBuildInputs = with ps.python3Packages; [
          gitpython
          langgraph
          langchain
          langchain-community
          langchain-openai
          nbclient
          pydantic
          typer
        ];
      };
  in
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = nixpkgs.lib.systems.flakeExposed;
      flake.overlays.default = final: prev: {
        expy = buildExpyFrom prev;
      };
      perSystem = {
        pkgs,
        system,
        ...
      }: let
        expy = buildExpyFrom pkgs;
        expyEditable = pkgs.python3.pkgs.mkPythonEditablePackage {
          pname = "expy";
          inherit (pyproject.project) scripts version;
          root = "$PKG_ROOT/src";
        };
      in {
        packages.default = expy;
        devShells.default = pkgs.mkShell {
          inputsFrom = [expy];
          buildInputs = with pkgs; [
            (pkgs.python3.withPackages (ps:
              with ps; [
                expyEditable
                ipython
                ipykernel
                jupyter
              ]))
          ];
          shellHook = ''
            python -m ipykernel install --user --name nix
            echo "warning: set PKG_ROOT env var for editable expy package to work"
          '';
        };
        apps.default = {
          type = "app";
          program = "${expy}/bin/expy";
        };
      };
    };
}
