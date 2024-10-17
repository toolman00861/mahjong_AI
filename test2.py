from mahjong_class import *

if __name__ == '__main__':
    mj = Mahjong_Manager()
    my_mahjong = ['wan_1', 'wan_2', 'wan_3', 'wan_4',
                  'tiao_1', 'tiao_7', 'tiao_8', 'tiao_9',
                  'tong_1', 'tong_1', 'tong_1', 'tong_2',
                  'zhong', 'zhong']
    mj.update_mahjong({}, my_mahjong)
    print(mj.calculate_mahjong())
    mahjong, score = mj.get_biggest_value_mahjong()
    print('推荐打：', mahjong, ' 价值：', score)
