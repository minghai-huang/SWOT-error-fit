%% SSHA error vs Wind Speed: geometry-aware, cubic-in-W, monotone-in-W fit (simplified)
% Keeps: 1, r^2, u^2, g1, g2   (drops: r^4, u^4)
clear; clc; close all

% ---------------------- Load ---------------------- %
% Expect: Pxxws.mat providing Pxxws(:,:,2) as [65 x numW] (cm)
% load Pxxws.mat
% Pxxws = Pxxws/sqrt(2);
% Pxxws  = double(Pxxws);

%% --------------------------------------- %%
% factor = sqrt(2.6)/sqrt(2);
%%
% % load Pxxswh.mat
% % Pxxswh = Pxxswh*factor;
% % SWH10_60km = SWH10_60km*factor;
% % load Pxxws.mat
% % Pxxws = Pxxws*factor;
% % WS10_60km = WS10_60km*factor;
%% --------------------------------------- %%
load Pxxws_0.13method.mat
Pxxws  = double(Pxxws);


nanflag = 31:35;                 % hide very center in plots if desired

% ---------------------- Data ---------------------- %
d  = (-64:2:64)';                % 65x1 km
W  = 4:2:16;                     % wind speed levels [m/s]
EE = Pxxws(:,:,2);               % error [cm], size = [65 x numel(W)]

% Beam-center (auto or fixed)
mcurve = mean(EE,2,'omitnan'); [~,i0] = min(mcurve); d0 = d(i0);
% d0 = 40;  % <- uncomment to force 40 km

% Nadir bumps
sigma1 = 12;                     % km (narrow) 0.081 cm
sigma2 = 22;                     % km (wide)
% sigma1 = 10;                     % km (narrow) RMSE=0.079 cm
% sigma2 = 27;                     % km (wide)

% Regularization / constraints
% lambda_list = [1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2];
lambda   = 1e-3;                 % ridge strength (1e-4–1e-2 good range)
eps_mono = 0.0;                  % monotone gap in W [cm]; e.g., 0.005 for visible separation
idx_mono = 1:numel(d);           % distances where monotonicity is enforced

% ---------------------- Design (no r^4, no u^4) ---------------------- %
[DD, WW] = ndgrid(d, W);
r  = abs(abs(DD) - d0);
u  = abs(DD);
g1 = exp(-(u./sigma1).^2);
g2 = exp(-(u./sigma2).^2);

% Vectorize
Y    = EE(:);
valid = ~isnan(Y);
rv   = r(:);   uv   = u(:);   g1v  = g1(:);   g2v  = g2(:);
WWv  = WW(:);

% Normalize W to [0,1]
Wmin = min(W); Wmax = max(W);
Wn = (WWv - Wmin) / (Wmax - Wmin);
w0 = ones(size(Wn));  w1 = Wn;  w2 = Wn.^2;  w3 = Wn.^3;

% Distance bases (keep: 1, r^2, u^2, g1, g2)
z1  = ones(size(Wn));
zr2 = rv.^2;
zu2 = uv.^2;

% Build X: 5 distance bases × 4 W-powers = 20 columns
X = [ ...
    z1 .*w0,  z1 .*w1,  z1 .*w2,  z1 .*w3, ...
    zr2.*w0,  zr2.*w1,  zr2.*w2,  zr2.*w3, ...
    zu2.*w0,  zu2.*w1,  zu2.*w2,  zu2.*w3, ...
    g1v.*w0,  g1v.*w1,  g1v.*w2,  g1v.*w3, ...
    g2v.*w0,  g2v.*w1,  g2v.*w2,  g2v.*w3 ];

% Apply NaN mask
X = X(valid,:);   Y = Y(valid);

% ---------------------- Monotone-in-wind constraints ---------------------- %
Aineq = []; bineq = [];
for ii = idx_mono(:).'
    for jj = 1:(numel(W)-1)
        r2 = basis_row_cubic_wind_simple(d(ii), W(jj+1), d0, sigma1, sigma2, Wmin, Wmax) ...
           - basis_row_cubic_wind_simple(d(ii), W(jj),   d0, sigma1, sigma2, Wmin, Wmax);   % 1×20
        % Enforce E(d, W_{j+1}) - E(d, W_j) >= eps_mono
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
eval_err = @(dvec,W0) local_eval_cubic_wind_simple(dvec,W0,d0,beta,sigma1,sigma2,Wmin,Wmax);
[DDg, WWg] = ndgrid(d, W);
Eg = reshape(eval_err(DDg(:), WWg(:)), size(DDg));   % 65×numW

% ---------------------- Plots ---------------------- %
co = lines(numel(W));
ix = (abs(d) >= 10 & abs(d) <= 60);
means_10_60 = mean(EE(ix,:), 1, 'omitnan');

% (1) Original (solid) vs Fitted (dashed)
figure('Units','normalized','Position',[0.12 0.12 0.56 0.72]); hold on
set(gcf,'Color','w'); grid on; box on
for j = 1:numel(W)
    plot(d, EE(:,j), 'o-', 'Color', co(j,:), 'MarkerSize',4, 'LineWidth',1.6, ...
        'DisplayName', sprintf('%dm/s; %.2fcm', W(j), means_10_60(j)));
    yj = Eg(:,j); if ~isempty(nanflag), yj(nanflag) = NaN; end
    plot(d, yj, '--', 'Color', co(j,:), 'LineWidth',2.1, 'HandleVisibility','off');
end
xlabel('Cross Track (km)'); ylabel('SSHA error (cm)')
title('SSHA error vs Wind — original vs fit (simplified)')
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
fprintf('Wind simplified model: K=%d  RMSE=%.3f cm  AIC=%.1f  BIC=%.1f\n', K, rmse, AIC, BIC);

% Optionally save fitted grid for later use
% save('Pxxws-fit-simplified.mat','Eg','d','W','nanflag','beta','d0','sigma1','sigma2','Wmin','Wmax');

% ---------------------- Export coefficients (4 significant figures) ----- %
thetaW = coeff_struct_wind_simple(beta);
Tw     = coeff_table_wind_simple(thetaW);
% A) numeric rounded values
Tw.Value = round(Tw.Value, 4, 'significant');
% writetable(Tw, 'wind_coeffs_simplified.csv', 'Encoding','UTF-8');

% % B) formatted strings with exactly 4 significant digits
% Tout = Tw;
% Tout.Value = compose('%.4g', Tw.Value);
% writetable(Tout, 'wind_coeffs_simplified.csv', 'Encoding','UTF-8');

%% ---------------------- Helpers (simplified basis) ---------------------- %%
function row = basis_row_cubic_wind_simple(dv, Wv, d0, sigma1, sigma2, Wmin, Wmax)
% One 1×20 row for (d,W) with cubic powers of normalized wind
    u = abs(dv);  r = abs(u - d0);
    g1 = exp(-(u/sigma1)^2);
    g2 = exp(-(u/sigma2)^2);
    Wn = (Wv - Wmin) / (Wmax - Wmin);
    svec = [1, Wn, Wn^2, Wn^3];             % 1×4
    z = [1, r^2, u^2, g1, g2];               % 1×5
    row = [];
    for k = 1:numel(z), row = [row, z(k)*svec]; end %#ok<AGROW>
end

function yf = local_eval_cubic_wind_simple(dvec,Wvec,d0,beta,sigma1,sigma2,Wmin,Wmax)
% Vectorized evaluation at arrays dvec,Wvec using simplified basis
    d1 = dvec(:);  W1 = Wvec(:);
    u1 = abs(d1);  r1 = abs(u1 - d0);
    g1 = exp(-(u1./sigma1).^2);
    g2 = exp(-(u1./sigma2).^2);
    Wn = (W1 - Wmin) ./ (Wmax - Wmin);
    w0 = ones(size(Wn)); w1 = Wn; w2 = Wn.^2; w3 = Wn.^3;
    z1 = ones(size(Wn));
    zr2 = r1.^2;
    zu2 = u1.^2;

    Xq = [ ...
      z1 .*w0,  z1 .*w1,  z1 .*w2,  z1 .*w3, ...
      zr2.*w0,  zr2.*w1,  zr2.*w2,  zr2.*w3, ...
      zu2.*w0,  zu2.*w1,  zu2.*w2,  zu2.*w3, ...
      g1 .*w0,  g1 .*w1,  g1 .*w2,  g1 .*w3, ...
      g2 .*w0,  g2 .*w1,  g2 .*w2,  g2 .*w3 ];
    yf = reshape(Xq*beta, size(dvec));
end

% ---------------------- Coefficient packaging (for paper) ---------------- %
function thetaW = coeff_struct_wind_simple(beta)
    beta = beta(:);
    % 5 groups × 4 W-powers = 20 elements
    thetaW.a = beta( 1: 4);   % baseline (1)
    thetaW.b = beta( 5: 8);   % r^2
    thetaW.c = beta( 9:12);   % u^2
    thetaW.d = beta(13:16);   % g1
    thetaW.e = beta(17:20);   % g2
end

function T = coeff_table_wind_simple(thetaW)
    groups = ["a (baseline)","b (r^2)","c (u^2)","d (g_1)","e (g_2)"];
    terms  = ["1","\\tilde W","\\tilde W^2","\\tilde W^3"];
    rows = {};
    vals = [];
    for gi = 1:numel(groups)
        coeffs = thetaW.(char('a'+gi-1));
        for ki = 1:4
            rows{end+1,1} = groups(gi); %#ok<AGROW>
            rows{end,  2} = sprintf('%s_%d', char('a'+gi-1), ki-1);
            rows{end,  3} = terms(ki);
            vals(end+1,1) = coeffs(ki); %#ok<AGROW>
        end
    end
    T = cell2table(rows, 'VariableNames', {'Group','Symbol','WTerm'});
    T.Value = vals;
end

% ---------------------- Standalone evaluator from packaged coeffs -------- %
function E = error_wind_E_simplified(d, W, thetaW, d0, sigma1, sigma2, Wmin, Wmax)
% Evaluate E(d,W) (cm) using grouped coefficients from coeff_struct_wind_simple
    u = abs(d); r = abs(u - d0);
    g1 = exp(-(u./sigma1).^2); g2 = exp(-(u./sigma2).^2);
    Wt = (W - Wmin) ./ (Wmax - Wmin);
    b = @(v) [1, v, v.^2, v.^3];     % [1, Wt, Wt^2, Wt^3]
    E = + (thetaW.a * b(Wt).').*1 ...
        + (thetaW.b * b(Wt).').*(r.^2) ...
        + (thetaW.c * b(Wt).').*(u.^2) ...
        + (thetaW.d * b(Wt).').*g1 ...
        + (thetaW.e * b(Wt).').*g2;
end
