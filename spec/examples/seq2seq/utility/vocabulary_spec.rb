require 'spec_helper'
require_relative '../../../../examples/seq2seq/utility/vocabulary'

describe Examples::Seq2seq::Utility::Vocabulary do
  let(:unknown_word) {Examples::Seq2seq::Utility::Vocabulary::UNK}
  let(:end_of_string) {Examples::Seq2seq::Utility::Vocabulary::EOS}

  let(:vocabulary) {Examples::Seq2seq::Utility::Vocabulary.new(dictionary)}
  let(:dictionary) {%w[ball paper cup]}

  context '#word_to_id' do
    it '辞書に存在する単語に対して正しいidが返ってくる' do
      expect(vocabulary.word_to_id('ball')).to eq 2
      expect(vocabulary.word_to_id('paper')).to eq 3
      expect(vocabulary.word_to_id('cup')).to eq 4
    end

    it '辞書に存在しない単語に対してUNK[:id]が返ってくる' do
      expect(vocabulary.word_to_id('hoge')).to eq unknown_word[:id]
    end

    it 'クラス定数で定義した単語に対して正しいidが返ってくる' do
      expect(vocabulary.word_to_id(unknown_word[:word])).to eq unknown_word[:id]
      expect(vocabulary.word_to_id(end_of_string[:word])).to eq end_of_string[:id]
    end
  end

  context '#id_to_word' do
    it '辞書に存在するidに対して正しい単語が返ってくる' do
      expect(vocabulary.id_to_word(2)).to eq 'ball'
      expect(vocabulary.id_to_word(3)).to eq 'paper'
      expect(vocabulary.id_to_word(4)).to eq 'cup'
    end

    it '辞書に存在しないidに対してUNK[:word]が返ってくる' do
      expect(vocabulary.id_to_word(99999)).to eq unknown_word[:word]
    end

    it 'クラス定数で定義したidに対して正しい単語が返ってくる' do
      expect(vocabulary.id_to_word(unknown_word[:id])).to eq unknown_word[:word]
      expect(vocabulary.id_to_word(end_of_string[:id])).to eq end_of_string[:word]
    end

    it '引数にInteger以外を渡すとTypeErrorが発生する' do
      expect{ vocabulary.id_to_word('1') }.to raise_error TypeError
    end
  end

  context '#size' do
    it '辞書の語数 + 2 が返ってくること' do
      expect(vocabulary.size).to eq dictionary.size + 2
    end
  end

  context '辞書内の単語が重複している場合' do
    let(:dictionary) {%w[ball paper paper cup]}

    context '#word_to_id' do
      it '辞書に存在する単語に対して正しいidが返ってくる' do
        expect(vocabulary.word_to_id('ball')).to eq 2
        expect(vocabulary.word_to_id('paper')).to eq 3
        expect(vocabulary.word_to_id('cup')).to eq 5
      end
    end

    context '#id_to_word' do
      it '辞書に存在するidに対して正しい単語が返ってくる' do
        expect(vocabulary.id_to_word(2)).to eq 'ball'
        expect(vocabulary.id_to_word(3)).to eq 'paper'
        expect(vocabulary.id_to_word(4)).to eq unknown_word[:word]
        expect(vocabulary.id_to_word(5)).to eq 'cup'
      end
    end

    context '#size' do
      it '辞書の語数 + 2 が返ってくること' do
        expect(vocabulary.size).to eq dictionary.uniq.size + 2
      end
    end
  end
end
