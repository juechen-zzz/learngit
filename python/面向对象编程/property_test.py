"""
    @property装饰器，slots限制类属性
"""


class Person(object):

    # 限定对象只能绑定_name. _age, _gender属性
    __slots__ = ('_name', '_age', '_gender')

    def __init__(self, name, age):
        self._name = name
        self._age = age

    # 访问器 - getter方法
    @property
    def name(self):
        return self._name

    # 访问器 - getter方法
    @property
    def age(self):
        return self._age

    # 修改器 - setter方法
    @age.setter
    def age(self, age):
        self._age = age

    def play(self):
        if self._age <= 16:
            print('%s在玩' % self._name)
        else:
            print('%s在学' % self._name)


def main():
    person = Person('xiaoming', 12)
    person.play()
    person.age = 22
    person.play()
    # person.name = '123'       AttributeError: can't set attribute
    person._gender = 'man'
    person._gender = 'woman'
    print(person._name)


if __name__ == '__main__':
    main()
