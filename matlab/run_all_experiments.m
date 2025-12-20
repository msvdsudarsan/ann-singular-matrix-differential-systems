function run_all_experiments()

    fprintf('=================================================\n');
    fprintf('PINN Framework for Singular Matrix Systems\n');
    fprintf('Reproducing Results from PDF Manuscript\n');
    fprintf('=================================================\n\n');

    %% ---------------- Problem 1 ----------------
    fprintf('\n>>> Running Problem 1: Singularly Perturbed BVP <<<\n');
    fprintf('=================================================\n');
    pinn_singular_perturbation();

    %% ---------------- Problem 2 ----------------
    fprintf('\n>>> Running Problem 2: Pantograph Delay Equation <<<\n');
    fprintf('=================================================\n');
    pinn_pantograph_delay();

    %% ---------------- Problem 3 ----------------
    fprintf('\n>>> Running Problem 3: Matrix Riccati Equation <<<\n');
    fprintf('=================================================\n');
    pinn_matrix_riccati();   % ✅ correct file & function name

    fprintf('\n=================================================\n');
    fprintf('ALL EXPERIMENTS COMPLETED SUCCESSFULLY\n');
    fprintf('=================================================\n');

end
