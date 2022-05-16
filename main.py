import jieba
import codecs
import pandas as pd
import numpy as np
from scipy.misc import imread
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt


# 读取聊天文本文件并分词
def get_chat_segment():
    # 加载自己的词典（可以定义聊天中的专属用语~）
    jieba.load_userdict("user_words.txt")

    # 打开聊天数据文件
    chat_file = codecs.open(u"chat_data.txt", 'r', encoding='utf-8')
    message = chat_file.read()
    chat_file.close()

    # 对整体进行分词，并保存分词结果
    segment = []
    raw_segment = jieba.cut(message)
    for seg in raw_segment:
        # 单字和换行符不加入数组
        if len(seg) > 1 and seg != '\r\n':
            segment.append(seg)

    return segment


# 获取词频字典
def get_words_dict():
    # 获得分词结果
    segment = get_chat_segment()
    df = pd.DataFrame({'segment': segment})

    # 加载停用词
    stopwords = pd.read_csv("stopwords.txt",
                            index_col=False,
                            names=['stopword'],
                            encoding="utf-8")

    # 如果不是在停用词中
    df = df[~df.segment.isin(stopwords.stopword)]

    # 按词分组，并计算每个词的词频
    words_count = df.groupby('segment')['segment'].agg(np.size).to_frame()
    words_count.columns = ['count']

    # 重排序，按照词频降序排列
    words_count = words_count.reset_index().sort_values(by="count",
                                                        ascending=False)

    return words_count


if __name__ == '__main__':
    # 获得词语和频数
    words_count = get_words_dict()

    # 读取我们想要生成词云的模板图片
    bg_img = imread('love.jpg')
    bg_img_colors = ImageColorGenerator(bg_img)

    # 获得词云对象，设定词云背景颜色、模板图片、大小、字体
    wordcloud = WordCloud(background_color='white',
                          mask=bg_img,
                          width=1200,
                          height=1000,
                          font_path='simhei.ttf')

    # 如果你的背景色是透明的，请用这两条语句替换上面两条
    # bg_img = imread('love.png')
    # wordcloud = WordCloud(background_color=None, mode='RGBA', mask=bg_img, width=1200, height=1000, font_path='simhei.ttf')

    # 将词语和频率转为字典
    words = words_count.set_index("segment").to_dict()

    # 将词语及频率映射到词云对象上
    wordcloud = wordcloud.fit_words(words["count"])

    # 将词云颜色重绘成模板图片颜色
    wordcloud.recolor(color_func=bg_img_colors)


    # plt.axis("off")
    # plt.imshow(wordcloud.recolor(color_func=bg_img_colors))
    # plt.savefig("output.png")

    # 保存至本地
    wordcloud.to_file("output.png")
    # plt.show()
