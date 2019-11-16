require 'csv'
require 'natto'
require_relative '../utility/eng'

module Dataset
  def self.get_jpn2eng(dataset_size=nil)
    dataset = []
    nm = Natto::MeCab.new('-Owakati')

    CSV.foreach('examples/seq2seq/data/jpn_eng_sentences.csv', col_sep: "\t", liberal_parsing: true) do |jpn_sentence, eng_sentence|
      jpn_sentence_words = nm.parse(jpn_sentence).split(' ')
      eng_sentence_words = Examples::Seq2seq::Utility::Eng.parse(eng_sentence + ' ').split(' ')

      dataset << [jpn_sentence_words, eng_sentence_words]

      break if dataset && dataset.size == dataset_size
    end

    dataset
  end
end
