require 'csv'
require 'natto'

jpn_words = ''
eng_words = ''

nm = Natto::MeCab.new('-Owakati')

CSV.foreach('examples/seq2seq/data/jpn_eng_sentences.csv', col_sep: "\t", liberal_parsing: true) do |row|
  jpn_sentence, eng_sentence = row
  jpn_words << nm.parse(jpn_sentence) + ' '
  eng_words << eng_sentence + ' '
end

jpn_dictionary = jpn_words.split(' ').uniq.join("\n")
eng_dictionary = eng_words
                   .delete("\",!?()")   # " , ! ? ( ) を消す
                   .gsub(/[\.]{3}/, '')             # 3点リード ... を消す
                   .gsub(/\.\s/, ' ')               # 単語末尾のピリオドを消す
                   .split(' ').uniq.join("\n")
                + "\n"
                + %w[" , . ... ! ? ( )].join("\n") # 消した記号たちを辞書に追加する

File.write('examples/seq2seq/data/jpn_dictionary.csv', jpn_dictionary)
File.write('examples/seq2seq/data/eng_dictionary.csv', eng_dictionary)
