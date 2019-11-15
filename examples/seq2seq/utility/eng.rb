module Examples
  module Seq2seq
    module Utility
      module Eng # :nodoc:
        def self.parse(text)
          text
            .gsub(/[\.]{2,}\s/, ' \0')
            .gsub(/(\w+?)\.\s/, '\1 . ')
        end
      end
    end
  end
end