
%% 截尾序贯抽样检测

% 设置随机种子确保结果可重复  
rng(42);
N = 10000;              % 总样本大小  
p0 = 0.1;              % 标称次品率  
data = rand(1, N) < p0;
%%
% 设置随机种子确保结果可重复 
clc;  
  

% N = 100;              % 总样本大小  
% p0 = 0.2;              % 标称次品率  
% data = rand(1, N) < p0; % 创建随机0-1序列  

% 定义参数  
max_rounds = 4;      % 最多检测轮数  
Z_alpha_95 = norminv(0.975);  % 95%信度的Z值  
Z_alpha_90 = norminv(0.975);   % 90%信度的Z值  



sample_size = 18;  % 从10到100的样本量  
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

        final_defect_rate_90 = n_total_90 / (round  *  sample_size ); % 95%信度下的次品率
       
 

% 输出95%信度下的结果  
fprintf('在95%%信度下接受的情况下，次品率: %.2f%%，\n最小检测次数: %d，最优样本量: %d，检测轮数: %d\n', ...  
        final_defect_rate_90 * 100, detection_count, sample_size, round);  

