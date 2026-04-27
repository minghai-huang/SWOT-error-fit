%% SSHA error vs SWH: geometry-aware, cubic-in-S, monotone-in-S fit (simplified)
% Keeps: 1, r^2, u^2, g1, g2   (drops: r^4, u^4)
clear; clc; close all

% ---------------------- Load ---------------------- %
% Expect: Pxxswh.mat providing Pxxswh(:,:,2) as [65 x numS] (cm)
% load Pxxswh.mat           % adjust if your filename differs
% Pxxswh = Pxxswh/sqrt(2);
% Pxxswh = double(Pxxswh);

%% --------------------------------------- %%
% factor = sqrt(2.63)/sqrt(2);

%%
% % load Pxxswh.mat
% % Pxxswh = Pxxswh*factor;
% % SWH10_60km = SWH10_60km*factor;
% % load Pxxws.mat
% % Pxxws = Pxxws*factor;
% % WS10_60km = WS10_60km*factor;
%% --------------------------------------- %%
load Pxxswh_0.13method.mat
Pxxswh = double(Pxxswh);



nanflag = 31:35;          % hide very center in plots if desired

% ---------------------- Data ---------------------- %
d  = (-64:2:64)';         % 65x1 km
S  = 1:1:6;               % SWH levels [m]
EE = Pxxswh(:,:,2);       % error [cm], size = [65 x numel(S)]

% Beam-center (auto or fixed)
mcurve = mean(EE,2,'omitnan'); [~,i0] = min(mcurve); d0 = d(i0);
% d0 = 40;  % <- uncomment to force 40 km

% Nadir bumps
sigma1 = 12;              % km (narrow) RMSE=0.043 cm
sigma2 = 22;              % km (wide)

% Regularization / constraints
lambda   = 1e-3;          % ridge strength (1e-4–1e-2 good range)
eps_mono = 0.0;           % monotone gap in S [cm]; e.g., 0.005 for visible separation
idx_mono = 1:numel(d);    % distances where monotonicity is enforced

% ---------------------- Design (no r^4, no u^4) ---------------------- %
[DD, SS] = ndgrid(d, S);
r  = abs(abs(DD) - d0);
u  = abs(DD);
g1 = exp(-(u./sigma1).^2);
g2 = exp(-(u./sigma2).^2);

% Vectorize
Y   = EE(:);
valid = ~isnan(Y);
rv  = r(:);   uv = u(:);   g1v = g1(:);   g2v = g2(:);
SSv = SS(:);

% Normalize S to [0,1]
Smin = min(S); Smax = max(S);
Sn = (SSv - Smin) / (Smax - Smin);
s0 = ones(size(Sn));  s1 = Sn;  s2 = Sn.^2;  s3 = Sn.^3;

% Distance bases (keep: 1, r^2, u^2, g1, g2)
z1  = ones(size(Sn));
zr2 = rv.^2;
zu2 = uv.^2;

% Build X: 5 distance bases × 4 S-powers = 20 columns
X = [ ...
    z1 .*s0,  z1 .*s1,  z1 .*s2,  z1 .*s3, ...
    zr2.*s0,  zr2.*s1,  zr2.*s2,  zr2.*s3, ...
    zu2.*s0,  zu2.*s1,  zu2.*s2,  zu2.*s3, ...
    g1v.*s0,  g1v.*s1,  g1v.*s2,  g1v.*s3, ...
    g2v.*s0,  g2v.*s1,  g2v.*s2,  g2v.*s3 ];

% Apply NaN mask
X = X(valid,:);   Y = Y(valid);

% ---------------------- Monotone-in-S constraints ---------------------- %
Aineq = []; bineq = [];
for ii = idx_mono(:).'
    for jj = 1:(numel(S)-1)
        r2 = basis_row_cubic_simple(d(ii), S(jj+1), d0, sigma1, sigma2, Smin, Smax) ...
           - basis_row_cubic_simple(d(ii), S(jj),   d0, sigma1, sigma2, Smin, Smax);   % 1×20
        % Enforce E(d, S_{j+1}) - E(d, S_j) >= eps_mono
        Aineq = [Aineq; -r2];                    %#ok<AGROW>
        bineq = [bineq; -eps_mono];              %#ok<AGROW>
    end
end

% ---------------------- Solve (ridge + constraints) ---------------------- %
ncoef  = size(X,2);                              % 20
X_aug  = [X; sqrt(lambda)*speye(ncoef)];
Y_aug  = [Y; zeros(ncoef,1)];

if exist('lsqlin','file') == 2
    opts = optimoptions('lsqlin','Display','off','MaxIter',5000, ...
        'OptimalityTolerance',1e-10,'ConstraintTolerance',1e-10);
    beta = lsqlin(X_aug, Y_aug, Aineq, bineq, [], [], [], [], [], opts);
else
    warning('lsqlin not found; solving unconstrained ridge system.');
    beta = (X_aug' * X_aug) \ (X_aug' * Y_aug);
end

% ---------------------- Evaluate on grid ---------------------- %
eval_err = @(dvec,S0) local_eval_cubic_simple(dvec,S0,d0,beta,sigma1,sigma2,Smin,Smax);
[DDg, SSg] = ndgrid(d, S);
Eg = reshape(eval_err(DDg(:), SSg(:)), size(DDg));   % 65×numS

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
    yj = Eg(:,j); if ~isempty(nanflag), yj(nanflag) = NaN; end
    plot(d, yj, '--', 'Color', co(j,:), 'LineWidth',2.1, 'HandleVisibility','off');
end
xlabel('Cross Track (km)'); ylabel('SSHA error (cm)')
title('SSHA error vs SWH — original vs fit (simplified)')
set(gca,'LineWidth',1.5,'FontSize',12)
xlim([min(d) max(d)]);
legend('Location','northeastoutside')
ylim([0 5]); grid on


% ---------------------- Fit metrics ---------------------- %
res  = EE - Eg;
rmse = sqrt(mean(res(~isnan(res)).^2));
N    = nnz(~isnan(EE));
RSS  = sum(res(~isnan(res)).^2);
K    = size(X,2);
AIC  = 2*K + N*log(RSS/N);
BIC  = K*log(N) + N*log(RSS/N);
fprintf('Simplified model: K=%d  RMSE=%.3f cm  AIC=%.1f  BIC=%.1f\n', K, rmse, AIC, BIC);

% Optionally save fitted grid for later use
% save('Pxxswh-fit-simplified.mat','Eg','d','S','nanflag','beta','d0','sigma1','sigma2','Smin','Smax');

% ---------------------- Export coefficients (4 significant figures) ----- %
thetaS = coeff_struct_swh(beta);
Tswh   = coeff_table_swh(thetaS);
% Keep numeric (rounded) OR write as formatted strings — choose ONE:
% A) numeric rounded values
Tswh.Value = round(Tswh.Value, 4, 'significant');
% writetable(Tswh, 'swh_coeffs_simplified.csv', 'Encoding','UTF-8');

% % B) formatted strings with exactly 4 significant digits
% Tout = Tswh;
% Tout.Value = compose('%.4g', Tswh.Value);
% writetable(Tout, 'swh_coeffs_simplified.csv', 'Encoding','UTF-8');

%% ---------------------- Helpers (simplified basis) ---------------------- %%
function row = basis_row_cubic_simple(dv, Sv, d0, sigma1, sigma2, Smin, Smax)
% One 1×20 row for (d,S) with cubic powers of normalized S
    u = abs(dv);  r = abs(u - d0);
    g1 = exp(-(u/sigma1)^2);
    g2 = exp(-(u/sigma2)^2);
    Sn = (Sv - Smin) / (Smax - Smin);
    svec = [1, Sn, Sn^2, Sn^3];               % 1×4
    z = [1, r^2, u^2, g1, g2];                 % 1×5
    row = [];
    for k = 1:numel(z), row = [row, z(k)*svec]; end %#ok<AGROW>
end

function yf = local_eval_cubic_simple(dvec,Svec,d0,beta,sigma1,sigma2,Smin,Smax)
% Vectorized evaluation at arrays dvec,Svec using simplified basis
    d1 = dvec(:);  S1 = Svec(:);
    u1 = abs(d1);  r1 = abs(u1 - d0);
    g1 = exp(-(u1./sigma1).^2);
    g2 = exp(-(u1./sigma2).^2);
    Sn = (S1 - Smin) ./ (Smax - Smin);
    s0 = ones(size(Sn)); s1 = Sn; s2 = Sn.^2; s3 = Sn.^3;
    z1 = ones(size(Sn));
    zr2 = r1.^2;
    zu2 = u1.^2;

    Xq = [ ...
      z1 .*s0,  z1 .*s1,  z1 .*s2,  z1 .*s3, ...
      zr2.*s0,  zr2.*s1,  zr2.*s2,  zr2.*s3, ...
      zu2.*s0,  zu2.*s1,  zu2.*s2,  zu2.*s3, ...
      g1 .*s0,  g1 .*s1,  g1 .*s2,  g1 .*s3, ...
      g2 .*s0,  g2 .*s1,  g2 .*s2,  g2 .*s3 ];
    yf = reshape(Xq*beta, size(dvec));
end

% ---------------------- Coefficient packaging (for paper) ---------------- %
function thetaS = coeff_struct_swh(beta)
    beta = beta(:);
    % 5 groups × 4 S-powers = 20 elements
    thetaS.a = beta( 1: 4);   % baseline (1)
    thetaS.b = beta( 5: 8);   % r^2
    thetaS.c = beta( 9:12);   % u^2
    thetaS.d = beta(13:16);   % g1
    thetaS.e = beta(17:20);   % g2
end

function T = coeff_table_swh(thetaS)
    groups = ["a (baseline)","b (r^2)","c (u^2)","d (g_1)","e (g_2)"];
    terms  = ["1","\tilde S","\tilde S^2","\tilde S^3"];
    rows = {};
    vals = [];
    for gi = 1:numel(groups)
        coeffs = thetaS.(char('a'+gi-1));
        for ki = 1:4
            rows{end+1,1} = groups(gi); %#ok<AGROW>
            rows{end,  2} = sprintf('%s_%d', char('a'+gi-1), ki-1);
            rows{end,  3} = terms(ki);
            vals(end+1,1) = coeffs(ki); %#ok<AGROW>
        end
    end
    T = cell2table(rows, 'VariableNames', {'Group','Symbol','STerm'});
    T.Value = vals;
end


