require 'csv'

sentences = {}
CSV.foreach('examples/seq2seq/data/sentences.csv', col_sep: "\t", liberal_parsing: true) do |row|
  id, language, sentence = row
  p "processing: #{id}" if (id.to_i % 10000).zero?
  next unless %w[jpn eng].include?(language)

  sentences[id] = sentence unless sentences.has_key?(id)
end

jpn_eng_sentences = ''
CSV.foreach('examples/seq2seq/data/jpn_indices.csv', col_sep: "\t", liberal_parsing: true) do |row|
  jpn_id, eng_id, other = row
  jpn_sentence = sentences[jpn_id]
  eng_sentence = sentences[eng_id]
  if jpn_sentence && eng_sentence
    jpn_eng_sentences << "#{jpn_sentence}\t#{eng_sentence}\n"
  end
end

File.write("examples/seq2seq/data/jpn_eng_sentences.csv", jpn_eng_sentences)
