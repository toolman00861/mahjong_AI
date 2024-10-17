# -*- coding: utf-8 -*-
# 日本麻将：牌类
# 分为万，筒，条，字牌
# 每个牌数量初始为4

Wan = ['wan_1', 'wan_2', 'wan_3', 'wan_4', 'wan_5', 'wan_6', 'wan_7', 'wan_8', 'wan_9']
Tong = ['tong_1', 'tong_2', 'tong_3', 'tong_4', 'tong_5', 'tong_6', 'tong_7', 'tong_8', 'tong_9']
Tiao = ['tiao_1', 'tiao_2', 'tiao_3', 'tiao_4', 'tiao_5', 'tiao_6', 'tiao_7', 'tiao_8', 'tiao_9']
Zi = ['dong', 'nan', 'xi', 'bei', 'zhong', 'fa', 'bai']

debug = False


# 2024.9。26
class Mahjong:
    def __init__(self, name, _type, num):
        self.name = name
        self.type = _type
        self.num = num

    def __str__(self):
        return f'{self.name} : {self.type} : {self.num}'


def get_mahjong_sym(mahjong):
    if mahjong.type == 'zi':
        # 抛出异常
        return 'error'
    return int(mahjong.name[-1:])


def get_mahjong_type(mahjong) -> str:
    if mahjong in Wan:
        return 'wan'
    if mahjong in Tong:
        return 'tong'
    if mahjong in Tiao:
        return 'tiao'
    if mahjong in Zi:
        return 'zi'


def can_be_shun_zi(a: Mahjong, b: Mahjong, c: Mahjong):
    """
    判断是否为顺子
    :param a: Mahjong
    :param b: Mahjong
    :param c: Mahjong
    :return: bool
    """
    if a.type == b.type == c.type:
        if a.type == 'zi':
            return False
        x = int(a.name[-1:])
        y = int(b.name[-1:])
        z = int(c.name[-1:])
        return x + 1 == y and y + 1 == z
    return False


def print_mahjong_list(mahjong_list):
    for mahjong in mahjong_list:
        print(mahjong.name, end=' ')
    print()


class Mahjong_Manager:
    def __init__(self):
        """
            初始化牌库，
            每次获取场上信息，都需要重置麻将管理器，重新计算权值。
        """

        self.my_mahjong = []  # 我的麻将
        self.mahjong_list = []  # 牌库中剩余的麻将
        for i in range(len(Wan)):
            self.mahjong_list.append(Mahjong(Wan[i], 'wan', 4))
            self.mahjong_list.append(Mahjong(Tong[i], 'tong', 4))
            self.mahjong_list.append(Mahjong(Tiao[i], 'tiao', 4))
        for i in range(len(Zi)):
            self.mahjong_list.append(Mahjong(Zi[i], 'zi', 4))

    def get_mahjong_list(self) -> list:
        return self.mahjong_list

    def update_mahjong(self, mahjong_on_field: dict, my_mahjong: list):
        """
        根据输入的list更新麻将库
        输入格式：
        {
            'wan_1': 1,
            'dong': 4,
            ....
            'tiao_5': 1
        }
        ['wan_1', 'tiao_1',...]
        :return: None
        """
        # 更新麻将库
        if debug:
            print('更新麻将库')
        for key, num in mahjong_on_field:
            for i in range(len(self.mahjong_list)):
                if num > 4:
                    print(f'输入的麻将数量有误！ --{key} : {num} 张')
                if self.mahjong_list[i].name == key:
                    self.mahjong_list[i].num -= num

        # 更新我的麻将
        if len(my_mahjong) <= 0 or len(my_mahjong) > 14:
            print('输入的麻将数量有误！')
            print(my_mahjong)
            return

        for mahjong in my_mahjong:
            _type = get_mahjong_type(mahjong)
            self.my_mahjong.append(Mahjong(mahjong, _type, 0))

        if debug:
            print('更新完成')

    def calculate_mahjong(self) -> dict:
        """
        枚举打出每张牌后，手牌的价值，给出一张表
        打出每张牌的价值比如：
        [
            {'w1' : 0.8}，
            {'w9' : 0.7}，
            ....
        ]
        :return:list
        """
        # v0.9 只考虑基胡
        mahjong_values = {}
        for mahjong_in_hand in self.my_mahjong:
            if debug:
                print('当前麻将：', mahjong_in_hand.name)
            name = mahjong_in_hand.name
            value = self.mahjong_list_value(mahjong_in_hand)
            mahjong_values[name] = value

        return mahjong_values

    def mahjong_list_value(self, mahjong_out) -> float:
        """
        2024.10.18更新
        打出某一麻将，计算胡牌距离，用于计算组成的牌价值
        :return: float
        """
        #
        num_pair = 0  # 对子
        num_set = 0  # 顺子或者刻字

        # 打出一张麻将后的牌形状
        mahjong_new = []
        find = False
        for mahjong in self.my_mahjong:
            if mahjong.name == mahjong_out.name and not find:
                find = True
                continue
            mahjong_new.append(mahjong)

        # 检查刻字或者顺子：
        # 刻子：
        map_ = {}
        for mahjong in mahjong_new:
            if mahjong.name not in map_:
                map_[mahjong.name] = 1
            else:
                map_[mahjong.name] += 1
        if debug:
            print(map_)
            print('麻将列表:', end=' ')
            print_mahjong_list(mahjong_new)
        for mahjong, num in map_.items():
            if num == 3:
                num_set += 1
                if debug:
                    print('移除麻将:', mahjong)
                while mahjong in mahjong_new:
                    for i in range(len(mahjong_new)):
                        if mahjong_new[i].name == mahjong:
                            del mahjong_new[i]
                            break
        # 顺子
        # 筛选出每个类别的牌
        mahjong_list_wan = []
        mahjong_list_tiao = []
        mahjong_list_tong = []
        for mahjong in mahjong_new:
            if mahjong.type == 'wan':
                mahjong_list_wan.append(mahjong)
            elif mahjong.type == 'tiao':
                mahjong_list_tiao.append(mahjong)
            elif mahjong.type == 'tong':
                mahjong_list_tong.append(mahjong)

        avail_signal = True  # 是否还有剩的
        while avail_signal:
            avail_signal = False
            for i in range(len(mahjong_list_wan)):
                if avail_signal:
                    break
                for j in range(i + 1, len(mahjong_list_wan)):
                    if avail_signal:
                        break
                    for k in range(j + 1, len(mahjong_list_wan)):
                        if can_be_shun_zi(mahjong_list_wan[i], mahjong_list_wan[j], mahjong_list_wan[k]):
                            num_set += 1
                            del mahjong_list_wan[k]
                            del mahjong_list_wan[j]
                            del mahjong_list_wan[i]
                            avail_signal = True
                            break

        avail_signal = True
        while avail_signal:
            avail_signal = False
            for i in range(len(mahjong_list_tiao)):
                if avail_signal:
                    break
                for j in range(i + 1, len(mahjong_list_tiao)):
                    if avail_signal:
                        break
                    for k in range(j + 1, len(mahjong_list_tiao)):
                        if can_be_shun_zi(mahjong_list_tiao[i], mahjong_list_tiao[j], mahjong_list_tiao[k]):
                            num_set += 1
                            del mahjong_list_tiao[k]
                            del mahjong_list_tiao[j]
                            del mahjong_list_tiao[i]
                            avail_signal = True
                            break

        avail_signal = True  # 是否还有剩的
        while avail_signal:
            avail_signal = False
            for i in range(len(mahjong_list_tong)):
                if avail_signal:
                    break
                for j in range(i + 1, len(mahjong_list_tong)):
                    if avail_signal:
                        break
                    for k in range(j + 1, len(mahjong_list_tong)):
                        if can_be_shun_zi(mahjong_list_tong[i], mahjong_list_tong[j], mahjong_list_tong[k]):
                            num_set += 1
                            del mahjong_list_tong[k]
                            del mahjong_list_tong[j]
                            del mahjong_list_tong[i]
                            avail_signal = True
                            break

        # 检查对子：
        for i in range(len(mahjong_new) - 1):
            if mahjong_new[i].name == mahjong_new[i + 1].name:
                num_pair += 1
                break

        # 检查剩余牌的距离
        dist = 0
        if len(mahjong_list_wan) >= 2:
            for i in range(len(mahjong_list_wan) - 1):
                a = get_mahjong_sym(mahjong_list_wan[i])
                b = get_mahjong_sym(mahjong_list_wan[i + 1])
                dist += abs(a - b)
        if len(mahjong_list_tiao) >= 2:
            for i in range(len(mahjong_list_tiao) - 1):
                a = get_mahjong_sym(mahjong_list_tiao[i])
                b = get_mahjong_sym(mahjong_list_tiao[i + 1])
                dist += abs(a - b)
        if len(mahjong_list_tong) >= 2:
            for i in range(len(mahjong_list_tong) - 1):
                a = get_mahjong_sym(mahjong_list_tong[i])
                b = get_mahjong_sym(mahjong_list_tong[i + 1])
                dist += abs(a - b)
        if debug:
            print('打出的牌：', mahjong_out.name, '剩余数量：', len(mahjong_new), '对子：', num_pair, '刻子：', num_set,
                  '距离：', dist)
            print('万牌：', end=' ')
            for mahjong in mahjong_list_wan:
                print(mahjong.name, end=' ')
            print()
            print('条牌：', end=' ')
            for mahjong in mahjong_list_tiao:
                print(mahjong.name, end=' ')
            print()
            print('筒牌：', end=' ')
            for mahjong in mahjong_list_tong:
                print(mahjong.name, end=' ')
            print()
        return 1 * num_pair + 1 * num_set / 4 + 1 / (dist + 1) * 1

    def get_mahjong_num(self, mahjong_name: str) -> int:
        """
        返回牌库中，某张牌的剩余数量
        :param mahjong_name:
        :return:
        """
        for mahjong in self.mahjong_list:
            if mahjong.name == mahjong_name:
                return mahjong.num

    def get_biggest_value_mahjong(self):
        """
        返回价值最大的牌，用于出牌
        :return:
        """
        score_dict = self.calculate_mahjong()
        max_score = 0
        res = None
        for mahjong in score_dict:
            if score_dict[mahjong] > max_score:
                max_score = score_dict[mahjong]
                res = mahjong
        return res, max_score
