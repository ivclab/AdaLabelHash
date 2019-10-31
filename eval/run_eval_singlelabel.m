
function run_eval_singlelabel(res_dir)

  database.label = readNPY(fullfile(res_dir, 'database_y.npy'));
  [max_vals, database.label] = max(database.label');
  database.label = database.label';

  query.label = readNPY(fullfile(res_dir, 'query_y.npy'));
  [max_vals, query.label] = max(query.label');
  query.label = query.label';

  database.feat = readNPY(fullfile(res_dir, 'database_x.npy'));
  query.feat = readNPY(fullfile(res_dir, 'query_x.npy'));

  database.binary = database.feat > 0.;
  query.binary = query.feat > 0.;

  map_file = fullfile(res_dir, 'map.txt');
  precision_file = fullfile(res_dir, 'precision-at-k.txt');
  pr_file = fullfile(res_dir, 'pr.txt');

  % retreival performance
  [map, precision_at_k, agg_prec] = precision_singlelabel(database.label, database.binary, ...
                                                          query.label, query.binary, 1);

  % save results
  outfile = fopen(map_file, 'w');
  fprintf(outfile, '%.4f\t', map);
  fclose(outfile);

  P = [[1:1:size(database.label,1)]' precision_at_k'];
  save(precision_file, 'P', '-ascii');

  pr = [[0:.1:1]' agg_prec'];
  save(pr_file, 'pr', '-ascii');
end
