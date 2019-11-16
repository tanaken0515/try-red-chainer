require 'spec_helper'
require_relative '../../../../examples/seq2seq/utility/eng'

describe Examples::Seq2seq::Utility::Eng do
  describe 'self.parse' do
    subject { Examples::Seq2seq::Utility::Eng.parse(text) }

    context 'ピリオドの扱い' do
      context '単語末尾に1つある場合' do
        let(:text) { 'I am fine. ' }
        it 'ピリオドが分割される' do
          expect(subject).to eq('I am fine . ')
        end
      end

      context '単語中にある場合' do
        let(:text) { 'Ruby 2.7.0 Preview 2 Released. ' }
        it '単語中のピリオドは分割されない' do
          expect(subject).to eq('Ruby 2.7.0 Preview 2 Released . ')
        end
      end

      context '連続している場合' do
        context '単語末尾の連続ピリオド' do
          let(:text) { 'Oh... so cool. ' }
          it '連続するピリオドを1単語として分割される' do
            expect(subject).to eq('Oh ... so cool . ')
          end
        end

        context '単語中の連続ピリオド' do
          let(:text) { 'Her shots are very fast but...a fast ball means that it will come back that much faster. ' }
          it '連続するピリオドを1単語として分割される' do
            expect(subject).to eq('Her shots are very fast but ... a fast ball means that it will come back that much faster . ')
          end
        end

        context '単語先頭の連続ピリオド' do
          let(:text) { ' ...Hey teacher. ' }
          it '連続するピリオドを1単語として分割される' do
            expect(subject).to eq(' ... Hey teacher . ')
          end
        end
      end
    end

    context 'スペースの扱い' do
      let(:text) { 'so    good' }
      it '連続するスペースは1つになる' do
        expect(subject).to eq('so good')
      end
    end

    context '! ? ( ) " , ; の扱い' do
      let(:text) { 'I said; "Oh..., so good!" (do you think so?) ' }
      it '分割される' do
        expect(subject).to eq('I said ; " Oh ... , so good ! " ( do you think so ? ) ')
      end

      context '数字の途中で , が使われている場合' do
        let(:text) { '3,000,000' }
        it '分割されない' do
          expect(subject).to eq('3,000,000')
        end
      end

      context '数字の末尾で , が使われている場合' do
        let(:text) { "Born on 2006 Feb 23, at 2730g, our family's treasure. " }
        it '分割される' do
          expect(subject).to eq("Born on 2006 Feb 23 , at 2730g , our family's treasure . ")
        end
      end

      context '記号の直前で , が使われている場合' do
        let(:text) { 'There\'s a saying, "once in a life-time event," and that\'s just what this is. ' }
        it '正しく , が分割される' do
          expect(subject).to eq('There\'s a saying , " once in a life-time event , " and that\'s just what this is . ')
        end
      end
    end

    context 'シングルクオーテーションの扱い' do
      let(:text) { "Please explain the grammar of 'as may be'. " }
      it '分割される' do
        expect(subject).to eq("Please explain the grammar of ' as may be ' . ")
      end

      context 'カンマの直後にシングルクォーテーションが来る場合' do
        let(:text) { "show the meaning of 'action, state, process, results,' and so on. " }
        it '分割される' do
          expect(subject).to eq("show the meaning of ' action , state , process , results , ' and so on . ")
        end
      end

      context 'カンマの直前にシングルクォーテーションが来る場合' do
        let(:text) { "What you call a 'charm', you know, is what girls wish upon that they may get together with the person they love and such like. " }
        it '分割される' do
          expect(subject).to eq("What you call a ' charm ' , you know , is what girls wish upon that they may get together with the person they love and such like . ")
        end
      end
    end
  end
end
