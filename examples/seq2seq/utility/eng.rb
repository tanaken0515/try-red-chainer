module Examples
  module Seq2seq
    module Utility
      module Eng # :nodoc:
        def self.parse(text)
          text.gsub(/\.\s/, ' . ')
        end
      end
    end
  end
end