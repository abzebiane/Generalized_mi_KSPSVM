function mi_kspsvm_tst(dataset_name)
%Generalized mi-KSPSVM Algorithm (MATLAB Implementation)
%Runs the generalized mi-KSPSVM with r=1,2,3 on a benchmark MIL dataset.
%
%Input:
%  dataset_name - string, name of the dataset (e.g., 'elephant', 'tst1')

%Load dataset
data_dir = '/home/anz325/qualifying_exam/final/';
load([data_dir dataset_name '.mat']);

bag_ids = full(bag_ids);
labels = full(labels);
features = double(features);

%Build bags from flat instance arrays
%Each bag is stored as a cell array of instance matrices
unique_bags = unique(bag_ids);
pos_bags = {};
neg_bags = {};

for i = 1:length(unique_bags)
    mask = bag_ids == unique_bags(i);
    instances = features(mask, :);
    instance_labels = labels(mask);
    if any(instance_labels == 1)
        pos_bags{end+1} = instances;   %Positive bag: at least one positive instance
    else
        neg_bags{end+1} = instances;   %Negative bag: all instances are negative
    end
end

fprintf('Dataset: %s\n', upper(dataset_name));
fprintf('Bags: %d positive, %d negative\n', length(pos_bags), length(neg_bags));

%Apply TruncatedSVD for dimensionality reduction on high-dimensional datasets
%Only applied when feature dimension exceeds 500 (e.g., TST text datasets)
all_instances = cell2mat(pos_bags');
all_instances = [all_instances; cell2mat(neg_bags')];
if size(all_instances, 2) > 500
    fprintf('Applying SVD dimensionality reduction...\n');
    [~, ~, V] = svds(sparse(all_instances), 200);   %Keep top 200 components
    for i = 1:length(pos_bags)
        pos_bags{i} = pos_bags{i} * V;
    end
    for i = 1:length(neg_bags)
        neg_bags{i} = neg_bags{i} * V;
    end
    fprintf('Reduced to 200 dimensions\n');
else
    fprintf('No dimensionality reduction needed (d=%d)\n', size(all_instances, 2));
end

%Initialize output file
output_file = [data_dir 'results_matlab_' dataset_name '.txt'];
fid = fopen(output_file, 'w');
fprintf(fid, 'Generalized mi-KSPSVM Results (MATLAB) - %s\n', upper(dataset_name));
fprintf(fid, 'Bi-level CV: C in 2^{-7..7}, sigma in 2^{-3,-2,2,3,4}\n');
fprintf(fid, '==================================================\n\n');
fclose(fid);

%Run for r = 1, 2, 3
for r = 1:3
    fprintf('\n--- r=%d ---\n', r);
    log_message(output_file, sprintf('\n--- r=%d ---', r));
    [acc, sens, spec, fs, cpu] = cross_validate(pos_bags, neg_bags, r, output_file);
    msg = sprintf('Accuracy: %.2f%% | Sensitivity: %.2f%% | Specificity: %.2f%% | F-score: %.2f%% | Avg Train Time: %.2fs', ...
        acc, sens, spec, fs, cpu);
    fprintf('%s\n', msg);
    log_message(output_file, msg);
end

fprintf('\nDone!\n');
log_message(output_file, 'Done!');
end


function [acc, sens, spec, fs, cpu] = cross_validate(pos_bags, neg_bags, r, output_file)
%10-fold cross-validation for the generalized mi-KSPSVM.
%
%The data is split into 10 equal folds. Each fold is used once as the
%test set while the remaining 9 folds form the training set.
%Hyperparameters C and sigma are selected via inner 5-fold CV on training data.
%Results are averaged across all folds.
%
%Returns: mean accuracy, sensitivity, specificity, F-score, avg CPU time

%Combine and shuffle all bags with their labels
all_bags = {};
for i = 1:length(pos_bags)
    all_bags{end+1} = {pos_bags{i}, 1};
end
for i = 1:length(neg_bags)
    all_bags{end+1} = {neg_bags{i}, -1};
end
n = length(all_bags);
idx = randperm(n);
all_bags = all_bags(idx);

n_folds = 10;
fold_size = floor(n / n_folds);
accuracies = []; sensitivities = []; specificities = []; fscores = []; train_times = [];

for fold = 1:n_folds
    %Split into train and test for this fold
    test_idx = (fold-1)*fold_size+1 : fold*fold_size;
    train_idx = setdiff(1:n, test_idx);
    test_set = all_bags(test_idx);
    train_set = all_bags(train_idx);

    %Separate positive and negative bags
    train_pos = get_bags(train_set, 1);
    train_neg = get_bags(train_set, -1);
    test_pos = get_bags(test_set, 1);
    test_neg = get_bags(test_set, -1);

    %Skip fold if any split is empty
    if isempty(train_pos) || isempty(train_neg), continue; end
    if isempty(test_pos) || isempty(test_neg), continue; end

    %Select best hyperparameters via inner 5-fold CV
    [best_C, best_sigma] = select_hyperparameters(train_pos, train_neg, r);
    msg = sprintf('  Fold %d: best_C=%.4f, best_sigma=%.4f', fold, best_C, best_sigma);
    fprintf('%s\n', msg);
    log_message(output_file, msg);

    %Train the generalized mi-KSPSVM and measure CPU time
    tic;
    [lambda_star, mu_star, X_plus, X_minus] = mi_kspsvm_alg(train_pos, train_neg, r, best_C, best_sigma);
    t = toc;
    train_times(end+1) = t;

    %Evaluate on test set
    [acc_f, sens_f, spec_f, fs_f] = evaluate(test_pos, test_neg, X_plus, X_minus, lambda_star, mu_star, best_sigma, r);
    accuracies(end+1) = acc_f;
    sensitivities(end+1) = sens_f;
    specificities(end+1) = spec_f;
    fscores(end+1) = fs_f;

    msg = sprintf('  Fold %d/10: accuracy=%.2f%%, train_time=%.2fs', fold, acc_f, t);
    fprintf('%s\n', msg);
    log_message(output_file, msg);
end

%Average results across all folds
acc = mean(accuracies); sens = mean(sensitivities);
spec = mean(specificities); fs = mean(fscores); cpu = mean(train_times);
end


function [best_C, best_sigma] = select_hyperparameters(train_pos, train_neg, r)
%Inner 5-fold cross-validation for hyperparameter selection.
%
%Searches over grids:
%  C     in {2^-7, 2^-6, ..., 2^7}
%  sigma in {2^-3, 2^-2, 2^2, 2^3, 2^4}
%
%Returns the (C, sigma) pair that maximizes average validation accuracy.

C_grid = 2.^(-7:7);
sigma_grid = 2.^[-3, -2, 2, 3, 4];
best_acc = -1; best_C = 1; best_sigma = 1;

%Combine training bags for inner CV
all_train = {};
for i = 1:length(train_pos)
    all_train{end+1} = {train_pos{i}, 1};
end
for i = 1:length(train_neg)
    all_train{end+1} = {train_neg{i}, -1};
end

n = length(all_train);
n_folds = 5;
fold_size = floor(n / n_folds);

for C = C_grid
    for sigma = sigma_grid
        fold_accs = [];
        for fold = 1:n_folds
            test_idx = (fold-1)*fold_size+1 : fold*fold_size;
            train_idx = setdiff(1:n, test_idx);
            inner_test = all_train(test_idx);
            inner_train = all_train(train_idx);

            ip = get_bags(inner_train, 1);
            in_ = get_bags(inner_train, -1);
            tp = get_bags(inner_test, 1);
            tn = get_bags(inner_test, -1);

            if isempty(ip) || isempty(in_) || isempty(tp) || isempty(tn), continue; end

            try
                [lam, mu, Xp, Xm] = mi_kspsvm_alg(ip, in_, r, C, sigma);
                [a, ~, ~, ~] = evaluate(tp, tn, Xp, Xm, lam, mu, sigma, r);
                fold_accs(end+1) = a;
            catch
                continue;
            end
        end

        %Update best hyperparameters if this combination performs better
        if ~isempty(fold_accs) && mean(fold_accs) > best_acc
            best_acc = mean(fold_accs);
            best_C = C; best_sigma = sigma;
        end
    end
end
end


function [lambda_star, mu_star, X_plus, X_minus] = mi_kspsvm_alg(pos_bags, neg_bags, r, C, sigma)
%Main iterative algorithm for the generalized mi-KSPSVM.
%
%Inputs:
%  pos_bags - cell array of positive bag instance matrices
%  neg_bags - cell array of negative bag instance matrices
%  r        - threshold parameter (r >= 1)
%  C        - regularization parameter
%  sigma    - RBF kernel bandwidth
%
%Returns:
%  lambda_star - optimal dual variables for J+
%  mu_star     - optimal dual variables for J-
%  X_plus      - final J+ instance matrix
%  X_minus     - final J- instance matrix (original negatives + moved instances)

%Stack all positive and negative instances into flat arrays
pos_instances = cell2mat(pos_bags');
neg_instances = cell2mat(neg_bags');

%Track which bag each positive instance belongs to
pos_bag_ids = [];
for i = 1:length(pos_bags)
    pos_bag_ids = [pos_bag_ids, repmat(i, 1, size(pos_bags{i}, 1))];
end

%Initialization: all positive instances in J+, J- contains only original negatives
J_plus = 1:size(pos_instances, 1);
J_minus_moved = [];   %Indices of positive instances moved to J-
m = length(pos_bags);

%Main iterative loop
while true
    %Build current J+ and J- instance matrices
    X_plus = pos_instances(J_plus, :);
    if isempty(J_minus_moved)
        X_minus = neg_instances;
    else
        X_minus = [neg_instances; pos_instances(J_minus_moved, :)];
    end

    %Compute kernel matrices for current J+ and J-
    Kpp = rbf_kernel(X_plus, X_plus, sigma);     %J+ vs J+
    Kmm = rbf_kernel(X_minus, X_minus, sigma);   %J- vs J-
    Kpm = rbf_kernel(X_plus, X_minus, sigma);    %J+ vs J-

    %Solve the kernelized dual problem to get optimal lambda and mu
    [lambda_star, mu_star] = solve_dual(Kpp, Kmm, Kpm, C);

    %Compute instance scores for all instances in J+
    %s_j = (K++ lambda* - K+- mu*)_j + e^T lambda* - e^T mu*
    scores = Kpp * lambda_star - Kpm * mu_star + sum(lambda_star) - sum(mu_star);

    %Compute J* (protected set): top-r instances per positive bag
    J_star = [];
    for i = 1:m
        %Find instances of bag i currently in J+
        bag_inst = find(pos_bag_ids(J_plus) == i);
        if isempty(bag_inst), continue; end

        %Sort by score descending and take top-r
        [~, sort_idx] = sort(scores(bag_inst), 'descend');
        top_r = bag_inst(sort_idx(1:min(r, end)));

        %Protect top-r instances if the worst among them scores <= -1
        if scores(top_r(end)) <= -1
            J_star = [J_star, top_r];
        end
    end

    %Compute J_bar: unprotected instances in J+ scoring <= -1
    unprotected = setdiff(1:length(J_plus), J_star);
    J_bar = unprotected(scores(unprotected) <= -1);

    %Termination: stop when no instances need to be moved
    if isempty(J_bar), break; end

    %Update J+ and J-: move J_bar instances from J+ to J-
    J_minus_moved = [J_minus_moved, J_plus(J_bar)];
    J_plus(J_bar) = [];
end
end


function [lambda_star, mu_star] = solve_dual(Kpp, Kmm, Kpm, C)
%Solve the kernelized Wolfe dual problem using quadprog.
%
%Dual problem:
%  min  0.5*lambda^T*(K++ + I/C + ee^T)*lambda
%       + 0.5*mu^T*(K-- + ee^T)*mu
%       + lambda^T*(-K+- - ee^T)*mu
%       - e^T*lambda - e^T*mu
%  s.t. 0 <= mu <= C*e
%  lambda unconstrained
%
%The constraint 0 <= mu <= C is encoded via lower/upper bounds in quadprog.
%
%Returns:
%  lambda_star - optimal dual variables for J+
%  mu_star     - optimal dual variables for J-

n_pos = size(Kpp, 1);
n_neg = size(Kmm, 1);
e_pos = ones(n_pos, 1);
e_neg = ones(n_neg, 1);

%Build QP matrix Q for z = [lambda; mu]
Q_pp = Kpp + eye(n_pos)/C + e_pos*e_pos';   %lambda-lambda block
Q_mm = Kmm + e_neg*e_neg';                   %mu-mu block
Q_pm = -Kpm - e_pos*e_neg';                  %lambda-mu cross block

Q = [Q_pp, Q_pm; Q_pm', Q_mm];

%Linear term: -e^T lambda - e^T mu
p = [-e_pos; -e_neg];

%Bounds: lambda unconstrained, 0 <= mu <= C
lb = [-inf(n_pos,1); zeros(n_neg,1)];
ub = [inf(n_pos,1); C*ones(n_neg,1)];

options = optimoptions('quadprog', 'Display', 'none');
z = quadprog(Q, p, [], [], [], [], lb, ub, [], options);

lambda_star = z(1:n_pos);
mu_star = z(n_pos+1:end);
end


function K = rbf_kernel(X, Y, sigma)
%Compute the RBF kernel matrix between rows of X and rows of Y.
%K[i,j] = exp(-||X[i] - Y[j]||^2 / (2*sigma^2))
%
%Inputs:
%  X     - (n1 x d) matrix
%  Y     - (n2 x d) matrix
%  sigma - bandwidth parameter
%
%Returns:
%  K - (n1 x n2) kernel matrix

X_norm = sum(X.^2, 2);
Y_norm = sum(Y.^2, 2);
dist_sq = bsxfun(@plus, X_norm, Y_norm') - 2*(X*Y');
dist_sq = max(dist_sq, 0);   %Prevent numerical errors from giving negative distances
K = exp(-dist_sq / (2*sigma^2));
end


function [acc, sens, spec, fs] = evaluate(pos_test, neg_test, X_plus, X_minus, lambda_star, mu_star, sigma, r)
%Evaluate classifier on test bags.
%
%Metrics:
%  tp: positive bags correctly predicted as positive
%  fn: positive bags wrongly predicted as negative
%  tn: negative bags correctly predicted as negative
%  fp: negative bags wrongly predicted as positive
%
%Returns: accuracy, sensitivity, specificity, F-score (all in %)

tp = 0; fn = 0;
for i = 1:length(pos_test)
    if predict_bag(pos_test{i}, X_plus, X_minus, lambda_star, mu_star, sigma, r) == 1
        tp = tp + 1;
    else
        fn = fn + 1;
    end
end

tn = 0; fp = 0;
for i = 1:length(neg_test)
    if predict_bag(neg_test{i}, X_plus, X_minus, lambda_star, mu_star, sigma, r) == -1
        tn = tn + 1;
    else
        fp = fp + 1;
    end
end

total = tp + fn + tn + fp;
acc = (tp + tn) / total * 100;
sens = 0; if (tp+fn) > 0, sens = tp/(tp+fn)*100; end
spec = 0; if (tn+fp) > 0, spec = tn/(tn+fp)*100; end
prec = 0; if (tp+fp) > 0, prec = tp/(tp+fp)*100; end
fs = 0; if (sens+prec) > 0, fs = 2*sens*prec/(sens+prec); end
end


function pred = predict_bag(bag, X_plus, X_minus, lambda_star, mu_star, sigma, r)
%Predict the label of a test bag.
%
%A bag is predicted positive if at least r of its instances score above 0.
%This is consistent with the threshold-based assumption used during training.
%
%Returns: 1 (positive) or -1 (negative)

K_plus = rbf_kernel(bag, X_plus, sigma);
K_minus = rbf_kernel(bag, X_minus, sigma);
b_star = sum(lambda_star) - sum(mu_star);
scores = K_plus * lambda_star - K_minus * mu_star + b_star;

if sum(scores > 0) >= r
    pred = 1;
else
    pred = -1;
end
end


function log_message(filepath, message)
%Append a message to the output results file.
fid = fopen(filepath, 'a');
fprintf(fid, '%s\n', message);
fclose(fid);
end


function bags = get_bags(bag_set, label)
%Extract bags of a given label from a combined bag set.
%
%Input:
%  bag_set - cell array of {instances, label} pairs
%  label   - 1 for positive, -1 for negative
%
%Returns:
%  bags - cell array of instance matrices for bags with the given label

bags = {};
for i = 1:length(bag_set)
    entry = bag_set{i};
    if entry{2} == label
        bags{end+1} = entry{1};
    end
end
end