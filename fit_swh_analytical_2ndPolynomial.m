%% SSHA error vs SWH: geometry-aware, quadratic-in-SWH, NO Gaussian bumps
% Keeps: 1, r^2, u^2
% Drops: g1, g2
% Uses:  [1, S~, S~^2]
clear; clc; close all

% ---------------------- Load ---------------------- %
% % Expect: Pxxswh.mat providing Pxxswh(:,:,2) as [65 x numS] (cm)
% % Adjust scaling below to match your current workflow
% % factor = sqrt(2.63)/sqrt(2);
% 
% % % load Pxxswh.mat
% % % Pxxswh    = Pxxswh * factor;
% % % SWH10_60km = SWH10_60km * factor;

load Pxxswh_0.13method.mat
Pxxswh = double(Pxxswh);

nanflag = 31:35;                 % hide very center in plots if desired

% ---------------------- Data ---------------------- %
d  = (-64:2:64)';                % 65x1 km
S  = 1:6;                        % SWH levels [m]
EE = Pxxswh(:,:,2);              % error [cm], size = [65 x numel(S)]

% Beam-center (auto or fixed)
mcurve = mean(EE,2,'omitnan');
[~,i0] = min(mcurve);
d0 = d(i0);
% d0 = 40;  % <- uncomment to force 40 km if desired

% Regularization / constraints
lambda   = 1e-3;                 % ridge strength
eps_mono = 0.0;                  % monotone gap in SWH [cm]
idx_mono = 1:numel(d);           % distances where monotonicity is enforced

% ---------------------- Design (NO Gaussian bumps) ---------------------- %
[DD, SS] = ndgrid(d, S);
r = abs(abs(DD) - d0);
u = abs(DD);

% Vectorize
Y     = EE(:);
valid = ~isnan(Y);
rv    = r(:);
uv    = u(:);
SSv   = SS(:);

% Normalize S to [0,1]
Smin = min(S);
Smax = max(S);
Sn   = (SSv - Smin) / (Smax - Smin);

s0 = ones(size(Sn));
s1 = Sn;
s2 = Sn.^2;

% Distance bases: 1, r^2, u^2
z1  = ones(size(Sn));
zr2 = rv.^2;
zu2 = uv.^2;

% Build X: 3 distance bases × 3 S-powers = 9 columns
X = [ ...
    z1 .*s0,  z1 .*s1,  z1 .*s2, ...
    zr2.*s0,  zr2.*s1,  zr2.*s2, ...
    zu2.*s0,  zu2.*s1,  zu2.*s2 ];

% Apply NaN mask
X = X(valid,:);
Y = Y(valid);

% ---------------------- Monotone-in-SWH constraints ---------------------- %
Aineq = [];
bineq = [];
for ii = idx_mono(:).'
    for jj = 1:(numel(S)-1)
        r2 = basis_row_quad_nobump(d(ii), S(jj+1), d0, Smin, Smax) ...
           - basis_row_quad_nobump(d(ii), S(jj),   d0, Smin, Smax);   % 1x9
        % Enforce E(d, S_{j+1}) - E(d, S_j) >= eps_mono
        Aineq = [Aineq; -r2]; %#ok<AGROW>
        bineq = [bineq; -eps_mono]; %#ok<AGROW>
    end
end

% ---------------------- Solve (ridge + constraints) ---------------------- %
ncoef = size(X,2);   % 9
X_aug = [X; sqrt(lambda)*speye(ncoef)];
Y_aug = [Y; zeros(ncoef,1)];

if exist('lsqlin','file') == 2
    opts = optimoptions('lsqlin','Display','off','MaxIter',5000, ...
        'OptimalityTolerance',1e-10,'ConstraintTolerance',1e-10);
    beta = lsqlin(X_aug, Y_aug, Aineq, bineq, [], [], [], [], [], opts);
else
    warning('lsqlin not found; solving unconstrained ridge system.');
    beta = (X_aug' * X_aug) \ (X_aug' * Y_aug);
end

% ---------------------- Evaluate on grid ---------------------- %
eval_err = @(dvec,S0) local_eval_quad_nobump(dvec,S0,d0,beta,Smin,Smax);
[DDg, SSg] = ndgrid(d, S);
Eg = reshape(eval_err(DDg(:), SSg(:)), size(DDg));   % 65 x numS

% ---------------------- Plots ---------------------- %
co = lines(numel(S));
ix = (abs(d) >= 10 & abs(d) <= 60);
means_10_60 = mean(EE(ix,:), 1, 'omitnan');

% (1) Original (solid) vs Fitted (dashed)
figure('Units','normalized','Position',[0.12 0.12 0.56 0.72]); hold on
set(gcf,'Color','w'); grid on; box on
for j = 1:numel(S)
    plot(d, EE(:,j), 'o-', 'Color', co(j,:), 'MarkerSize',4, 'LineWidth',1.6, ...
        'DisplayName', sprintf('%dm; %.2fcm', S(j), means_10_60(j)));
    yj = Eg(:,j);
    if ~isempty(nanflag), yj(nanflag) = NaN; end
    plot(d, yj, '--', 'Color', co(j,:), 'LineWidth',2.1, 'HandleVisibility','off');
end
xlabel('Cross Track (km)');
ylabel('SSHA error (cm)');
title('SSHA error vs SWH — original vs fit (quadratic, no bumps)')
set(gca,'LineWidth',1.5,'FontSize',12)
xlim([min(d) max(d)]);
ylim([0 5]);
legend('Location','northeastoutside')


% ---------------------- Fit metrics ---------------------- %
res  = EE - Eg;
rmse = sqrt(mean(res(~isnan(res)).^2));
N    = nnz(~isnan(EE));
RSS  = sum(res(~isnan(res)).^2);
K    = size(X,2);
AIC  = 2*K + N*log(RSS/N);
BIC  = K*log(N) + N*log(RSS/N);

fprintf('SWH quadratic no-bump model: K=%d  RMSE=%.4f cm  AIC=%.1f  BIC=%.1f\n', ...
    K, rmse, AIC, BIC);

% ---------------------- Optional save ---------------------- %
% save('Pxxswh-fit-quad-nobump.mat', 'Eg','d','S','nanflag','beta','d0','Smin','Smax');

% ---------------------- Export coefficients ---------------------- %
thetaS = coeff_struct_swh_quad_nobump(beta);
Ts     = coeff_table_swh_quad_nobump(thetaS);

% numeric rounded values
Ts.Value = round(Ts.Value, 4, 'significant');

% writetable(Ts, 'swh_coeffs_quad_nobump.csv', 'Encoding','UTF-8');

%% ---------------------- Helpers ---------------------- %%
function row = basis_row_quad_nobump(dv, Sv, d0, Smin, Smax)
% One 1x9 row for (d,S) with quadratic powers of normalized SWH
    u = abs(dv);
    r = abs(u - d0);
    Sn = (Sv - Smin) / (Smax - Smin);

    svec = [1, Sn, Sn^2];      % 1x3
    z = [1, r^2, u^2];         % 1x3

    row = [];
    for k = 1:numel(z)
        row = [row, z(k)*svec]; %#ok<AGROW>
    end
end

function yf = local_eval_quad_nobump(dvec,Svec,d0,beta,Smin,Smax)
% Vectorized evaluation at arrays dvec,Svec using quadratic no-bump basis
    d1 = dvec(:);
    S1 = Svec(:);

    u1 = abs(d1);
    r1 = abs(u1 - d0);
    Sn = (S1 - Smin) ./ (Smax - Smin);

    s0 = ones(size(Sn));
    s1 = Sn;
    s2 = Sn.^2;

    z1  = ones(size(Sn));
    zr2 = r1.^2;
    zu2 = u1.^2;

    Xq = [ ...
      z1 .*s0,  z1 .*s1,  z1 .*s2, ...
      zr2.*s0,  zr2.*s1,  zr2.*s2, ...
      zu2.*s0,  zu2.*s1,  zu2.*s2 ];

    yf = reshape(Xq*beta, size(dvec));
end

% ---------------------- Coefficient packaging ---------------------- %
function thetaS = coeff_struct_swh_quad_nobump(beta)
    beta = beta(:);
    % 3 groups x 3 S-powers = 9 elements
    thetaS.a = beta(1:3);   % baseline (1)
    thetaS.b = beta(4:6);   % r^2
    thetaS.c = beta(7:9);   % u^2
end

function T = coeff_table_swh_quad_nobump(thetaS)
    groups = ["a (baseline)","b (r^2)","c (u^2)"];
    terms  = ["1","\tilde S","\tilde S^2"];

    rows = {};
    vals = [];
    for gi = 1:numel(groups)
        coeffs = thetaS.(char('a'+gi-1));
        for ki = 1:3
            rows{end+1,1} = groups(gi); %#ok<AGROW>
            rows{end,  2} = sprintf('%s_%d', char('a'+gi-1), ki-1);
            rows{end,  3} = terms(ki);
            vals(end+1,1) = coeffs(ki); %#ok<AGROW>
        end
    end
    T = cell2table(rows, 'VariableNames', {'Group','Symbol','STerm'});
    T.Value = vals;
end

% ---------------------- Standalone evaluator ---------------------- %
function E = error_swh_E_quad_nobump(d, S, thetaS, d0, Smin, Smax)
% Evaluate E(d,S) (cm) using grouped coefficients
    u  = abs(d);
    r  = abs(u - d0);
    St = (S - Smin) ./ (Smax - Smin);
    b  = @(v) [1, v, v.^2];

    E = + (thetaS.a * b(St).') .* 1 ...
        + (thetaS.b * b(St).') .* (r.^2) ...
        + (thetaS.c * b(St).') .* (u.^2);
end



