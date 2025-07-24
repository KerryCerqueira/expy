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
				buildExpyFrom = ps: ps.python3Packages.buildPythonPackage {
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
				expy = buildExpyFrom pkgs;
				expyEditable = pkgs.python3.pkgs.mkPythonEditablePackage {
					pname = "expy";
					inherit (pyproject.project) scripts version;
					root = "$PKG_ROOT/src";
				};
			in {
				overlays.default = final: prev: {
					expy = buildExpyFrom prev;
				};
				packages.default = expy;
				devShells = {
					default = pkgs.mkShell {
						inputsFrom = [ expy ];
						buildInputs = with pkgs; [
							(python3.withPackages ( ps: with ps; [
								expyEditable
								ipython
								ipykernel
								jupyter
							]))
						];
						shellHook = '' python -m ipykernel install --user --name nix
						echo "warning: set PKG_ROOT env var for editable expy package to work"
						'';
					};
				};
			}
		);
}
