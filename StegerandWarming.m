% This code computes the numerical solution of the
% shock tube problem. The flow has zero initial velocity.
% The input data are:
% prat: pressure ratio
% denrat: density ratio
% time: time instant at which solution is computed
% N: number of equally spaced points.
% Code written by Jo Wayne Tan, 01327317
% Dept. of Aeronautics, Imperial College, January 2021.

clear
clc
close all

tic

%% Initialise variables
mesh = [100 200 300]; % Different meshes
rho_output = zeros(3, 300); % (3 x 300) matrices with 300 being the largest mesh to store all domain values
p_output = zeros(3, 300);
u_output = zeros(3, 300);
Mach_output = zeros(3, 300);
ent_output = zeros(3, 300);

for n = 1:length(mesh)

    % Enter the values of prat, denrat, time and N here
    prat = 10;   % pressure ratio
    denrat = 8;  % density ratio
    time = 0.5;  % time to compute the solution
    N = mesh(n); % number of equally spaced points in the interval

    % Set x interval
    xmin = -2.;
    xmax = 2.;
    xlen = xmax - xmin;
    dx = xlen/N; % Delta x
    x = linspace(xmin, xmax, N); % Discritised space

    %% Initial Conditions, initially at rest
    gamma = 1.4;
    % Driver part (2) with driver air
    p2 = prat;
    rho2 = denrat;
    u2 = 0;
    E2 = p2/((gamma - 1)*rho2) + (u2^2)/2;
    H2 = E2 + p2/rho2;
    
    % Driven part (1) with driven air
    p1 = p2/prat;       % p_1 = 1
    rho1 = rho2/denrat; % rho_1 = 1
    u1 = 0;
    E1 = p1/((gamma - 1)*rho1) + (u1^2)/2;
    H1 = E1 + p1/rho1;

    % Set values in the domain
    p = appendvalues(p1, p2, N); % pressure
    rho = appendvalues(rho1, rho2, N); % density
    u = appendvalues(u1, u2, N); % velocity
    E = appendvalues(E1, E2, N); % energy
    H = appendvalues(H1, H2, N); % enthalpy
    c = sqrt(gamma*abs(p./rho)); % Local speed of sound

    % Initiate solution vector U at time level n and over the entire domain (i=1:N)
    U = zeros(3, N); % Initialise (3 x N) matrix
    U(1,:) = rho;
    U(2,:) = rho.*u;
    U(3,:) = rho.*E;
    
    % Initialise solution vector U_new at time level n+1 over the entire domain (i=1:N)
    U_new = zeros(3,N);

    %% Steger and Warming Flux Vector Splitting Method
    % Eigenvalues at time, t = 0
    lambda = zeros(3,N);
    lambda_plus = zeros(3,N);
    lambda_minus = zeros(3,N);

    % Fluxes at time, t = 0
    F_plus = zeros(3,N);
    F_minus = zeros(3,N);
    
    t = 0; % Set time to zero
    % t increased by value of timestep until it reaches the solution time
    while t < time
        for i = 1:N
            lambda(:,i) = [u(i)-c(i); u(i); u(i)+c(i)]; % store only non-zero values of the diagonal eigenvalue matrix
            % Split eigenvalues
            lambda_plus(:,i) = 0.5*(lambda(:,i) + abs(lambda(:,i)));
            lambda_minus(:,i) = 0.5*(lambda(:,i) - abs(lambda(:,i)));

            % Positive flux, F+, from coursework handout equation (10)
            F_plus(:,i) = (rho(i)/(2*gamma))*[lambda_plus(1,i) + 2*(gamma-1)*lambda_plus(2,i) + lambda_plus(3,i);...
                          (u(i)-c(i))*lambda_plus(1,i) + 2*(gamma-1)*u(i)*lambda_plus(2,i) + (u(i)+c(i))*lambda_plus(3,i);...
                          (H(i)-(u(i)*c(i)))*lambda_plus(1,i) + (gamma-1)*((u(i))^2)*lambda_plus(2,i) + (H(i)+(u(i)*c(i)))*lambda_plus(3,i)];

            % Negative flux, F-, from coursework handout equation (10)
            F_minus(:,i) = (rho(i)/(2*gamma))*[lambda_minus(1,i) + 2*(gamma-1)*lambda_minus(2,i) + lambda_minus(3,i);...
                           (u(i)-c(i))*lambda_minus(1,i) + 2*(gamma-1)*u(i)*lambda_minus(2,i) + (u(i)+c(i))*lambda_minus(3,i);...
                           (H(i)-(u(i)*c(i)))*lambda_minus(1,i) + (gamma-1)*((u(i))^2)*lambda_minus(2,i) + (H(i)+(u(i)*c(i)))*lambda_minus(3,i)];

        end
        
        % Maximum timestep for stability. Only need to find max of third row of lambda_plus because 
        % lambda_minus contains u - c values and second row of lambda_plus
        % contains u values hence will always be smaller than u + c assuming u > 0
        dt = 0.98*dx/max(lambda_plus(3,:));
        
        % Evaluate solution at timestep t^n+1
        % Boundary condition at x = -2, U_0 = U_1 hence F+_0 = F+_1
        U_new(:,1) = U(:,1) - (dt/dx)*((F_plus(:,1) - F_plus(:,1)) + (F_minus(:,2) - F_minus(:,1)));
        
        % First order upwind scheme
        for j = 2:N-1
            U_new(:,j) = U(:,j) - (dt/dx)*((F_plus(:,j) - F_plus(:,j-1)) + (F_minus(:,j+1) - F_minus(:,j)));
        end
        
        % Boundary condition at x = 2, U_{N+1} = U_{N} hence F-_{N+1} = F-_{N}
        U_new(:,N) = U(:,N) - (dt/dx)*((F_plus(:,N) - F_plus(:,N-1)) + (F_minus(:,N) - F_minus(:,N)));
        
        % Replace U with U_new after each iteration and update all variables
        U = U_new;
        rho = U(1,:);
        u = U(2,:)./U(1,:);
        E = U(3,:)./U(1,:);
        p = (gamma - 1).*rho.*(E - 0.5*u.^2);
        H = E + p./rho;
        c = sqrt(gamma*abs(p./rho));
        t = t + dt; % Increment by current timestep to get next time level
        
    end
    
    %% Compile results
    ent = log(p./(rho.^gamma)); % Entropy
    M = u./c; % Mach number
    
    for k = 1:N
        rho_output(n,k) = rho(k);
        u_output(n,k) = u(k);
        p_output(n,k) = p(k);
        Mach_output(n,k) = M(k);
        ent_output(n,k) = ent(k);
    end
    
end

% Output values to txt format and plot in live script
writematrix(rho_output, 'rho.txt')
writematrix(u_output, 'u.txt')
writematrix(p_output, 'p.txt')
writematrix(Mach_output, 'Mach.txt')
writematrix(ent_output, 'entropy.txt')

toc

% Function to append variable values to parts (1) and (2)
function [var] = appendvalues(var1, var2, N)
    var = zeros(1, N);
    var(ceil((N + 1)/2):N) = var1; % Part (1)
    var(1:ceil((N + 1)/2)) = var2; % Part (2)
end
