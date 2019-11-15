module Examples
  module Seq2seq
    module Utility
      module Eng # :nodoc:
        def self.parse(text)
          text
            .gsub(/["!\(\)\?]/, ' \0 ')
            .gsub(/([^\d])(,)([^\d])/, '\1 \2 \3')
            .gsub(/(\d)(,)(\s)/, '\1 \2 \3')
            .gsub(/[\.]{2,}\s/, ' \0')
            .gsub(/\s[\.]{2,}/, '\0 ')
            .gsub(/(\w+?)([\.]{2,})(\w+?)/, '\1 \2 \3')
            .gsub(/(\w+?)\.\s/, '\1 . ')
            .gsub(/(')(\.\s)/, '\1 \2')
            .gsub(/(\s')(.+?)('\s)/, '\1 \2 \3')
            .gsub(/ {1,}/, ' ')
        end
      end
    end
  end
end