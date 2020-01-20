from string import ascii_lowercase

!split -a 3 -l 100000 '/content/drive/My Drive/en.txt' '/content/project/lines/lines-'

for c1 in ascii_lowercase:
  for c2 in ascii_lowercase:
    script = '/content/drive/My\ Drive/nlp_project/create_data.py'
    output_dir = '/content/drive/My\ Drive/nlp_project/conversational_JSON_ALL_small/'+c1+c2+'/'
    sentence_files_var = '/content/drive/My\ Drive/nlp_project/lines/lines-' + c1 + c2 +'\*'
    !python {script} --sentence_files {sentence_files_var} --output_dir {output_dir} --runner DirectRunner --temp_location /content/project/temp --staging_location /content/project/staging --project nlp --dataset_format JSON