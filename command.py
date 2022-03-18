import subprocess

magnolia = 'cabal exec -v0 magnolia --'

def run_compiler(rewrite_confs):
    package_name = "examples.pde.mg-src.pde-cpp"
    command_line = [magnolia, "build"]
    command_line += ["-o", "examples/pde/cpp-src/gen"]
    command_line += ["--backend", "cpp"]
    command_line += ["--base-import-directory", "gen"]
    command_line += ["--allow-overwrite"]
    command_line.append("examples/pde/mg-src/pde-cpp.mg")
    command_line += ["--programs-to-rewrite",
                     f"{package_name}.PDEProgram"]
    command_line += ["--rewriting-system-configs",
                     "'" + ','.join(f"{package_name}.{name}|{steps}" for name,steps in rewrite_confs) + "'"]

    command_line = [' '.join(command_line)]
    process = subprocess.run(command_line, capture_output=True, shell=True)
    print(process.returncode)
    print(process.stderr.decode('utf-8'))
    print(process.stdout.decode('utf-8'))

if __name__ == '__main__':
    rew_configs = [ ("DNFRules", 100)
                  , ("OFRavel", 1)
                  , ("ShapeIsUnique", 100)
                  , ("ONFToArrayRules", 100)
                  , ("PsiCorrespondenceTheorem", 1)
                  , ("ShapeIsUnique", 100) # Could rewrite next rewrite rules
                  #, ("OFTile", 1)
                  #, ("OFLiftCores", 1)
                  ##, ("OFIntroducePaddingInArguments", 1)
                  #, ("OFIntroducePaddingRule", 1)
                  ##, ("OFAddLeftPadding0Axis", 10)
                  ##, ("OFAddRightPadding0Axis", 10)
                  #, ("OFExtractInnerRule", 1)
                  #, ("OFRemoveLeftoverPadding", 100)
                  ]
    run_compiler(rew_configs)
