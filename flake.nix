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
				shellHook = # bash
					''
				python -m ipykernel install --user --name nix
				'';
			in { devShells = {
					cpuInference = let
						pkgs = import nixpkgs {
							system = system;
							config.allowUnfree = true;
						};
					in pkgs.mkShell {
							buildInputs = with pkgs; [
								(python3.withPackages ( ps: with ps; [
									gitpython
									torch
									transformers
									llama-cpp-python
									ipython
									ipykernel
									langchain
									langchain-community
									langchain-huggingface
									langgraph
									nbclient
									pydantic
								]))
								jupyter
							];
							inherit shellHook;
						};
					cudaInference = let
						pkgs = import nixpkgs {
							system = system;
							config.allowUnfree = true;
							config.cudaSupport = true;
						};
					in pkgs.mkShell {
							buildInputs = with pkgs; [
								(python3.withPackages ( ps: with ps; [
									gitpython
									torch
									transformers
									llama-cpp-python
									ipython
									ipykernel
									langchain
									langchain-community
									langchain-huggingface
									langgraph
									nbclient
									pydantic
								]))
								jupyter
							];
							LD_PRELOAD = "/usr/lib/x86_64-linux-gnu/libcuda.so.1";
							inherit shellHook;
						};
				};
			}
		);
}
