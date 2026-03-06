%% 
%% Paper Title: "Adaptive Physics-Informed Neural Networks for Singular Matrix
%%               Differential Systems with Applications to Optimal Control Synthesis"
%% Author 1:    Sri Venkata Durga Sudarsan Madhyannapu
%% Author 2:    Pradheep Kumar S.
%%
%% Affiliation 1: Freshmen Engineering Department, NRI Institute of Technology
%%                (Autonomous), Pothavarappadu, Agiripalli, Eluru District 521212,
%%                Andhra Pradesh, India
%% Affiliation 2: Research Scholar, Jawaharlal Nehru Technological University
%%                Kakinada, Andhra Pradesh, India
%% Affiliation 3: School of Basic Sciences, SRM University AP, Neerukonda,
%%                Mangalagiri, Guntur-522240, Andhra Pradesh, India
%%
%% Journal:       Neurocomputing (Elsevier), ISSN: 0925-2312
%% Manuscript ID: NEUCOM-D-26-03849
%% Status:        With Editor, 2026
%% SSRN:          https://ssrn.com/abstract=6277631
%% SSRN ID:       6277631 (Distributed: 02/20/2026)
%%
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
