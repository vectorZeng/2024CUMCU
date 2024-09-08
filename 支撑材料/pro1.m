
%% 截尾序贯抽样检测

% 设置随机种子确保结果可重复  
rng(4);
N = 10000;              % 总样本大小  
p0 = 0.05;              % 标称次品率  
data = rand(1, N) < p0;
%%
% 设置随机种子确保结果可重复 
clc;  


results_95 = [];       % 存储95%信度下的次品数量  
results_90 = [];       % 存储90%信度下的次品数量  
% N = 100;              % 总样本大小  
p0 = 0.1;              % 标称次品率  
% data = rand(1, N) < p0; % 创建随机0-1序列  

% 定义参数  
max_rounds = 100;      % 最多检测轮数  
Z_alpha_95 = norminv(0.975);  % 95%信度的Z值  
Z_alpha_90 = norminv(0.975);   % 90%信度的Z值  

% (1) 在95%信度下拒收的情况下，寻找最小检测次数  
min_detection_count_95 = inf; % 初始化最小检测次数  
optimal_sample_size_95 = 0;    % 最优样本量  

for sample_size = 10:100 % 从10到100的样本量  
    results_95 = []; 
    n_total_95 = 0;  % 95%信度下的次品总数  
    accepted_95 = false; % 95%信度下是否接受  

    for round = 1:max_rounds  
        if (round - 1) * sample_size + sample_size > N  
            break; % 确保不会超出总样本数  
        end  
        indices = randperm(N, sample_size); % 生成随机索引  
        sample = data(indices); % 根据随机索引抽样  
        % sample = data((round - 1) * sample_size + 1: round * sample_size); % 抽样  
        n = sum(sample);  % 次品数量  
        n_total_95 = n_total_95 + n;  % 统计95%信度下的次品总数  

        % 更新均值和标准差  
        mean_95 = n_total_95 / round;  
        std_dev_95 = std([results_95, n]);  % 次品数量的标准差  

        % 计算接受范围  
        L_95 = mean_95 - Z_alpha_95 * (std_dev_95 / sqrt(round));  
        U_95 = mean_95 + Z_alpha_95 * (std_dev_95 / sqrt(round));  

        % 检测接受与否  
        if n > U_95  
            accepted_95 = false;  % 拒绝这批零配件  
            break;  % 停止抽样  
        elseif n < L_95  
            accepted_95 = true;  % 接受这批零配件 
            
            break;  % 停止抽样  
        
        else
        
        results_95(end+1) = n;  % 继续抽样  
        end
    end  

    % 计算检测次数  
    detection_count = round * sample_size;  

    if ~accepted_95 && detection_count < min_detection_count_95  
        min_detection_count_95 = detection_count; % 更新最小检测次数  
        optimal_sample_size_95 = sample_size;     % 更新最优样本量 
        round_95 = round;
        n_total_95_min = n_total_95;
        % 计算95%信度下的次品率  
        final_defect_rate_95 = n_total_95_min / (round_95  * sample_size); % 95%信度下的次品率  
    end  
end  



% 输出95%信度下的结果  
fprintf('在95%%信度下拒收的情况下，次品率: %.2f%%，\n最小检测次数: %d，最优样本量: %d，检测轮数: %d\n', ...  
        final_defect_rate_95 *100 , min_detection_count_95, optimal_sample_size_95, round_95);  


% 
% % 输出95%信度下的结果  
% fprintf('95%%信度下的检测轮数: %d\n', round);  
% fprintf('95%%信度下的最终次品数量: %d\n', n_total_95);   
% fprintf('95%%信度下的次品率: %.2f%%\n', final_defect_rate_95 / 100);  

% (2) 在90%信度下接受的情况下，寻找最小检测次数  
min_detection_count_90 = inf; % 初始化最小检测次数  
optimal_sample_size_90 = 0;    % 最优样本量  

for sample_size = 10:100 % 从10到100的样本量  
    results_90 = []; 
    n_total_90 = 0;  % 90%信度下的次品总数  
    accepted_90 = false; % 90%信度下是否接受  

    for round = 1:max_rounds  
        if (round - 1) * sample_size + sample_size > N  
            break; % 确保不会超出总样本数  
        end  

        indices = randperm(N, sample_size); % 生成随机索引  
        sample = data(indices); % 根据随机索引抽样  
        % sample = data((round - 1) * sample_size + 1: round * sample_size); % 抽样  
        n = sum(sample);  % 次品数量  
        n_total_90 = n_total_90 + n;  % 统计90%信度下的次品总数  

        % 更新均值和标准差  
        mean_90 = n_total_90 / round; 
        
        results_90(end+1) = n;  % 继续抽样  
        std_dev_90 = std(results_90);  % 次品数量的标准差  

        % 计算接受范围  
        L_90 = mean_90 - Z_alpha_90 * (std_dev_90 / sqrt(round));  
        U_90 = mean_90 + Z_alpha_90 * (std_dev_90 / sqrt(round));  

        % 检测接受与否  
        if n < L_90  
            accepted_90 = true;  % 接受这批零配件  
            

            break;  % 停止抽样  
        elseif n > U_90  
            accepted_90 = false;  % 拒绝这批零配件  
            
            break;  % 停止抽样  
         
        
        end
    end  

    % 计算检测次数  
    detection_count = round * sample_size;  

    if accepted_90 && detection_count < min_detection_count_90  
        min_detection_count_90 = detection_count; % 更新最小检测次数  
        optimal_sample_size_90 = sample_size;     % 更新最优样本量  
        round_90 = round;
        n_total_90_min = n_total_90;
        % 计算95%信度下的次品率  
        final_defect_rate_90 = n_total_90_min / (round_90  * optimal_sample_size_90); % 95%信度下的次品率
    end  
end  



% 输出95%信度下的结果  
fprintf('在90%%信度下接受的情况下，次品率: %.2f%%，\n最小检测次数: %d，最优样本量: %d，检测轮数: %d\n', ...  
        final_defect_rate_90 * 100, min_detection_count_90, optimal_sample_size_90, round_90);  


%%
% 初始化参数  
N = 10000;              % 总样本大小  
num_samples = 100;      % 每个比例的样本数量  
p_defect_ratios = 0:0.01:1;  % 次品比例从0到1  
acceptance_probabilities = zeros(size(p_defect_ratios)); % 接受概率  

% 对于每个次品比例进行抽样  
for i = 1:length(p_defect_ratios)  
    p0 = p_defect_ratios(i); % 当前次品比例  
    data = rand(1, N) < p0;  % 根据比例生成次品数据  

    accepted_count = 0; % 被接受的次数  

    for j = 1:num_samples  
        sample_size = 100; % 每次抽样的样本量  
        indices = randperm(N, sample_size); % 随机抽样  
        sample = data(indices); % 抽样数据  
        n = sum(sample);  % 次品数量  

        % 决策逻辑（假设的范围，可以根据您的模型调整）  
        if n < sample_size * 0.05 % 设定的接受条件  
            accepted_count = accepted_count + 1; % 接受  
        end  
    end  

    % 计算接受概率  
    acceptance_probabilities(i) = accepted_count / num_samples; % 接受概率  
end  

% 绘制 OC 曲线  
figure;  
plot(p_defect_ratios, acceptance_probabilities, '-','LineWidth',1.5);  
xlabel('失效概率');  
ylabel('接受概率');  
legend('操作特征曲线 (OC Curve)')
% title('操作特征曲线 (OC Curve)');  
grid on;  

% 显示额外信息  
xlim([0 0.9]);  
ylim([0 1]);
set(gca,'FontSize',25)

% 保存图形为 .eps 文件  
print('oc_curve', '-depsc'); % 'oc_curve.eps' 将被创建  