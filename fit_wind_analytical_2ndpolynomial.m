%% SSHA error vs Wind Speed: geometry-aware, quadratic-in-W, NO Gaussian bumps
% Keeps: 1, r^2, u^2
% Drops: g1, g2
% Uses:  [1, W~, W~^2]
clear; clc; close all

% ---------------------- Load ---------------------- %
% Expect: Pxxws.mat providing Pxxws(:,:,2) as [65 x numW] (cm)
% Adjust scaling below to match your current workflow
% factor = sqrt(2.63)/sqrt(2);

% load Pxxws.mat
% Pxxws = Pxxws * factor;
% % WS10_60km = WS10_60km * factor;   % uncomment if needed


load Pxxws_0.13method.mat
Pxxws = double(Pxxws);

nanflag = 31:35;                 % hide very center in plots if desired

% ---------------------- Data ---------------------- %
d  = (-64:2:64)';                % 65x1 km
W  = 4:2:16;                     % wind speed levels [m/s]
EE = Pxxws(:,:,2);               % error [cm], size = [65 x numel(W)]

% Beam-center (auto or fixed)
mcurve = mean(EE,2,'omitnan');
[~,i0] = min(mcurve);
d0 = d(i0);
% d0 = 40;  % <- uncomment to force 40 km if desired

% Regularization / constraints
lambda   = 1e-3;                 % ridge strength
eps_mono = 0.0;                  % monotone gap in wind speed [cm]
idx_mono = 1:numel(d);           % distances where monotonicity is enforced

% ---------------------- Design (NO Gaussian bumps) ---------------------- %
[DD, WW] = ndgrid(d, W);
r = abs(abs(DD) - d0);
u = abs(DD);

% Vectorize
Y     = EE(:);
valid = ~isnan(Y);
rv    = r(:);
uv    = u(:);
WWv   = WW(:);

% Normalize W to [0,1]
Wmin = min(W);
Wmax = max(W);
Wn   = (WWv - Wmin) / (Wmax - Wmin);

w0 = ones(size(Wn));
w1 = Wn;
w2 = Wn.^2;

% Distance bases: 1, r^2, u^2
z1  = ones(size(Wn));
zr2 = rv.^2;
zu2 = uv.^2;

% Build X: 3 distance bases × 3 W-powers = 9 columns
X = [ ...
    z1 .*w0,  z1 .*w1,  z1 .*w2, ...
    zr2.*w0,  zr2.*w1,  zr2.*w2, ...
    zu2.*w0,  zu2.*w1,  zu2.*w2 ];

% Apply NaN mask
X = X(valid,:);
Y = Y(valid);

% ---------------------- Monotone-in-Wind constraints ---------------------- %
Aineq = [];
bineq = [];
for ii = idx_mono(:).'
    for jj = 1:(numel(W)-1)
        r2 = basis_row_quad_nobump_wind(d(ii), W(jj+1), d0, Wmin, Wmax) ...
           - basis_row_quad_nobump_wind(d(ii), W(jj),   d0, Wmin, Wmax);   % 1x9
        % Enforce E(d, W_{j+1}) - E(d, W_j) >= eps_mono
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
eval_err = @(dvec,W0) local_eval_quad_nobump_wind(dvec,W0,d0,beta,Wmin,Wmax);
[DDg, WWg] = ndgrid(d, W);
Eg = reshape(eval_err(DDg(:), WWg(:)), size(DDg));   % 65 x numW

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
    yj = Eg(:,j);
    if ~isempty(nanflag), yj(nanflag) = NaN; end
    plot(d, yj, '--', 'Color', co(j,:), 'LineWidth',2.1, 'HandleVisibility','off');
end
xlabel('Cross Track (km)');
ylabel('SSHA error (cm)');
title('SSHA error vs Wind Speed — original vs fit (quadratic, no bumps)')
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

fprintf('Wind quadratic no-bump model: K=%d  RMSE=%.4f cm  AIC=%.1f  BIC=%.1f\n', ...
    K, rmse, AIC, BIC);

% ---------------------- Optional save ---------------------- %
save('Pxxws-fit-quad-nobump.mat', 'Eg','d','W','nanflag','beta','d0','Wmin','Wmax');

% ---------------------- Export coefficients ---------------------- %
thetaW = coeff_struct_wind_quad_nobump(beta);
Tw     = coeff_table_wind_quad_nobump(thetaW);

% numeric rounded values
Tw.Value = round(Tw.Value, 4, 'significant');

% writetable(Tw, 'wind_coeffs_quad_nobump.csv', 'Encoding','UTF-8');

%% ---------------------- Helpers ---------------------- %%
function row = basis_row_quad_nobump_wind(dv, Wv, d0, Wmin, Wmax)
% One 1x9 row for (d,W) with quadratic powers of normalized wind
    u = abs(dv);
    r = abs(u - d0);
    Wn = (Wv - Wmin) / (Wmax - Wmin);

    wvec = [1, Wn, Wn^2];      % 1x3
    z = [1, r^2, u^2];         % 1x3

    row = [];
    for k = 1:numel(z)
        row = [row, z(k)*wvec]; %#ok<AGROW>
    end
end

function yf = local_eval_quad_nobump_wind(dvec,Wvec,d0,beta,Wmin,Wmax)
% Vectorized evaluation at arrays dvec,Wvec using quadratic no-bump basis
    d1 = dvec(:);
    W1 = Wvec(:);

    u1 = abs(d1);
    r1 = abs(u1 - d0);
    Wn = (W1 - Wmin) ./ (Wmax - Wmin);

    w0 = ones(size(Wn));
    w1 = Wn;
    w2 = Wn.^2;

    z1  = ones(size(Wn));
    zr2 = r1.^2;
    zu2 = u1.^2;

    Xq = [ ...
      z1 .*w0,  z1 .*w1,  z1 .*w2, ...
      zr2.*w0,  zr2.*w1,  zr2.*w2, ...
      zu2.*w0,  zu2.*w1,  zu2.*w2 ];

    yf = reshape(Xq*beta, size(dvec));
end

% ---------------------- Coefficient packaging ---------------------- %
function thetaW = coeff_struct_wind_quad_nobump(beta)
    beta = beta(:);
    % 3 groups x 3 W-powers = 9 elements
    thetaW.a = beta(1:3);   % baseline (1)
    thetaW.b = beta(4:6);   % r^2
    thetaW.c = beta(7:9);   % u^2
end

function T = coeff_table_wind_quad_nobump(thetaW)
    groups = ["a (baseline)","b (r^2)","c (u^2)"];
    terms  = ["1","\tilde W","\tilde W^2"];

    rows = {};
    vals = [];
    for gi = 1:numel(groups)
        coeffs = thetaW.(char('a'+gi-1));
        for ki = 1:3
            rows{end+1,1} = groups(gi); %#ok<AGROW>
            rows{end,  2} = sprintf('%s_%d', char('a'+gi-1), ki-1);
            rows{end,  3} = terms(ki);
            vals(end+1,1) = coeffs(ki); %#ok<AGROW>
        end
    end
    T = cell2table(rows, 'VariableNames', {'Group','Symbol','WTerm'});
    T.Value = vals;
end

% ---------------------- Standalone evaluator ---------------------- %
function E = error_wind_E_quad_nobump(d, W, thetaW, d0, Wmin, Wmax)
% Evaluate E(d,W) (cm) using grouped coefficients
    u  = abs(d);
    r  = abs(u - d0);
    Wt = (W - Wmin) ./ (Wmax - Wmin);
    b  = @(v) [1, v, v.^2];

    E = + (thetaW.a * b(Wt).') .* 1 ...
        + (thetaW.b * b(Wt).') .* (r.^2) ...
        + (thetaW.c * b(Wt).') .* (u.^2);
end

