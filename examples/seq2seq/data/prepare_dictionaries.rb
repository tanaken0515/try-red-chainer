require 'csv'

jpn_words = ''
eng_words = ''

CSV.foreach('examples/seq2seq/data/jpn_eng_sentences.csv', col_sep: "\t", liberal_parsing: true) do |row|
  jpn_sentence, eng_sentence = row
  eng_words << eng_sentence + ' '
end

jpn_dictionary = ''
eng_dictionary = eng_words.split(' ').uniq.join("\n")
File.write('examples/seq2seq/data/jpn_dictionary.csv', jpn_dictionary)
File.write('examples/seq2seq/data/eng_dictionary.csv', eng_dictionary)
