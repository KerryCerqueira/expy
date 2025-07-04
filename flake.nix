{
	description = "Reproducible AI experiments with python";
	inputs = {
		nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
		flake-utils .url = "github:numtide/flake-utils";
	};

	outputs =
		{ nixpkgs, flake-utils, ... }:
		flake-utils.lib.eachDefaultSystem (system:
			let
				pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
				pkgs = import nixpkgs {
					system = system;
				};
				expy = pkgs.python3Packages.buildPythonApplication {
					pname = "expy";
					inherit (pyproject.project) version;
					format = "pyproject";
					src = ./.;
					build-system = with pkgs.python3Packages; [
						setuptools
					];
					propagatedBuildInputs = with pkgs.python3Packages; [
						gitpython
						langgraph
						nbclient
						pydantic
						typer
					];
				};
				expy-dev = pkgs.python3.pkgs.mkPythonEditablePackage {
					pname = "expy";
					inherit (pyproject.project) scripts version;
					root = "$PWD/src";
				};
			in {
				packages = {
					inherit expy;
				};
				devShells = {
					default = pkgs.mkShell {
						inputsFrom = [ expy ];
						buildInputs = with pkgs; [
							expy-dev
							(python3.withPackages ( ps: with ps; [
								ipython
								ipykernel
								langchain
								langchain-community
								langchain-huggingface
							]))
							jupyter
						];
						shellHook = # bash
							''
							python -m ipykernel install --user --name nix
							'';
					};
				};
			}
		);
}
