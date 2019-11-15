module Examples
  module Seq2seq
    module Utility
      module Eng # :nodoc:
        def self.parse(text)
          text
            .gsub(/[",!\(\)\?]/, ' \0 ')
            .gsub(/[\.]{2,}\s/, ' \0')
            .gsub(/\s[\.]{2,}/, '\0 ')
            .gsub(/(\w+?)\.\s/, '\1 . ')
            .gsub(/ {1,}/, ' ')
        end
      end
    end
  end
end