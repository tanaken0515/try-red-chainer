require 'csv'

module Examples
  module Seq2seq
    module Utility
      class Vocabulary
        UNK = {id: 0, word: '<UNK>'}
        EOS = {id: 1, word: '<EOS>'}

        def initialize(dictionary)
          set_w2i_i2w(dictionary)
        end

        def word_to_id(word)
          @w2i[word] || UNK[:id]
        end

        def id_to_word(id)
          @i2w[id] || UNK[:word]
        end

        def size
          @w2i.size
        end

        def self.from_csv(file_path)
          dictionary = CSV.read(file_path, liberal_parsing: true).flatten
          self.new(dictionary)
        end

        private

        def set_w2i_i2w(dictionary)
          @w2i = {UNK[:word] => UNK[:id], EOS[:word] => EOS[:id]}
          @i2w = []
          @i2w[UNK[:id]] = UNK[:word]
          @i2w[EOS[:id]] = EOS[:word]

          dictionary.each_with_index do |word, i|
            unless @w2i.has_key?(word)
              id = i + 2
              @w2i[word] = id
              @i2w[id] = word
            end
          end
        end
      end
    end
  end
end
