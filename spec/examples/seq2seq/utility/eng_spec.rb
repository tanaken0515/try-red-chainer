require 'spec_helper'
require_relative '../../../../examples/seq2seq/utility/eng'

describe Examples::Seq2seq::Utility::Eng do
  describe 'self.parse' do
    subject { Examples::Seq2seq::Utility::Eng.parse(text) }

    context '単語末尾にピリオドがある場合' do
      let(:text) { 'I am fine. ' }
      it 'ピリオドが分割される' do
        expect(subject).to eq('I am fine . ')
      end
    end
  end
end
