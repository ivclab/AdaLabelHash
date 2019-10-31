function run_eval(code_lens)
  if nargin == 0
    code_lens = {12, 24, 32, 48};
  end

  current_abspath = pwd;
  [root_path, ~, ~] = fileparts(current_abspath);

  exp_path = fullfile(root_path, 'experiments/cifar10_supB')
  src_path = fullfile(root_path, 'eval');

  cd(src_path);

  for i = 1:length(code_lens)
    run_eval_singlelabel(fullfile(exp_path, ['models/' int2str(code_lens{i}) 'bits']))
  end

  cd(current_abspath);
end
