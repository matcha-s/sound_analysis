from pyfilterbank import FractionalOctaveFilterbank
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os
import csv
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description='This program performs octave band analysis. (オクターブ解析をするプログラム)')
    parser.add_argument('input', nargs='*', help='The path name of audio file to input. (入力する音声ファイルのフォルダ名(1つ)またはパス名(複数))')
    parser.add_argument('-o', '--output', help='The path name of image file of the graph to be output. (出力するグラフの画像ファイルのパス名)')
    parser.add_argument('-c', '--csv', help='The path name of csv file of analysis results to be output. (出力する分析結果のCSVファイルのパス名)')
    parser.add_argument('-l', '--linear', action='store_false', help='With this option, the graph will be scaled linearly. (指定した場合、線形スケールに変更する)')
    parser.add_argument('-n', '--nth', default=3.0, type=float, help="How many times should you filter it? (default : 3) (何分の一オクターブバンド解析をするか(デフォルトは 3 ))")
    parser.add_argument('-f', '--filter', default=4, type=int, help="The number of filter order (default:4) (フィルター次数(デフォルトは 4 ))")
    parser.add_argument('-r', '--index_remove', action='store_false', help='With this option, remove index from the graph. (指定した場合、グラフから索引を削除する)')
    parser.add_argument('-i', '--index_list', nargs='+', help='Index the input audio files in the specified order. (入力された音声ファイルの順に指定された索引をつける)' )
    parser.add_argument('-s', '--dont_show', action='store_false', help='With this option, Hide the window of the graph. (指定した場合、グラフのウィンドウを非表示)')

    args = parser.parse_args()

    if len(args.input) == 1 and os.path.isdir(str(args.input[0])):
        ext = ('wav', 'mp3')
        inputs = []
        for e in ext:
            inputs += glob.glob(os.path.join(args.input[0], '*.' + e))
    
    else:
        inputs = args.input

    if args.index_list == None :
        index_list = [ os.path.basename(f) for f in inputs ]
    
    elif args.index_list != None and len(inputs) != len(args.index_list):
        try:
            raise Exception("The number of input audio files and the number in index_list do not match. (入力された音声ファイルの数と引数の索引の数が一致しない)")
        except Exception as e:
            print(str(e))
            print("This program use file name as index.(ファイル名を索引として使用)")
            index_list = [ os.path.basename(f) for f in inputs ]
    
    else:
        index_list = args.index_list

    analyse_octave_band(inputs, nth_oct=args.nth, order=args.filter, x_log=args.linear, graphpath=args.output, index=args.index_remove, label_list=index_list, csvpath=args.csv, showgraph=args.dont_show)



def analyse_octave_band(filepaths, nth_oct = 3.0, order = 4, x_log = True, graphpath = None, index = True, label_list = [], csvpath=None, showgraph = True):

    temp = 0
    for f in filepaths:

        # オーディオファイルの読み込み
        wave, sr = sf.read(f)

        # 1 / nth_oct オクターブバンド分析
        ofb = FractionalOctaveFilterbank(sample_rate = sr, order = order, nth_oct = nth_oct)
        signal, states = ofb.filter(wave)

        # 各バンドのエネルギーをdB単位に変換
        L_sum = np.sum(signal ** 2, axis = 0)
        np.seterr(divide='ignore')
        L = 10 * np.log10( L_sum / np.max(L_sum) )
        
        # 各バンドの中心周波数
        freqs = np.array(list(states.keys()))

        # プロット
        plt.grid(True)
        if label_list:
            plt.plot(freqs, L, label = label_list[temp])
            
            if csvpath != None and not os.path.isfile(csvpath):
                with open(csvpath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["freqs"] + freqs.tolist())
                
            if csvpath != None:
                with open(csvpath, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([label_list[temp]] + L.tolist())
        
        else:
            plt.plot(freqs, L, label = os.path.basename(os.path.dirname(f)))

            if csvpath != None and not os.path.isfile(csvpath):
                with open(csvpath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([""] + freqs.tolist())
                
            if csvpath != None:
                with open(csvpath, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(os.path.basename(os.path.dirname(f)) + L.tolist())

        temp += 1

    # 
    if x_log == True:
        plt.xscale('log')
    plt.title("Octave-band analysis")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("SPL [dB]")
    label = [ 10, 100, 1000, 10000 ] 
    plt.xticks(label, map(str, label))
    if index == True:
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol = len(filepaths))
    if graphpath != None:
        plt.savefig(graphpath, bbox_inches='tight')
    if showgraph == True:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()