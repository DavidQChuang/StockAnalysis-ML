import math


class TradingSimulation:
    def __init__(self, money) -> None:
        self.money: float = money
        self.starting_money: float = money
        self.volume: float = 0

    def valuation(self, current_price):
        return self.money + self.volume * current_price

    def market_valuation(self, starting_price, current_price):
        return self.starting_money / starting_price * current_price

    def step(self, action, current_price):
        """Performs the given action (0b01 to sell, 0b10 to buy, 0b11 for both, 0b00 for none)
        then returns the previous valuation.
        """
        prev_value = self.valuation(current_price)
        # sell
        if action & 1 != 0:
            shares = self.volume

            self.money += current_price * shares
            self.volume = 0

        # buy
        if action & 2 != 0:
            shares = int(self.money / current_price)

            self.money -= current_price * shares
            self.volume += shares

        return prev_value

    def scale_money(self, money):
        return 2/(1 + math.exp(20*(1 - money/self.starting_money))) - 1
    
    def scale_delta(self, delta):
        return 2/(1 + math.exp(20*(-delta/self.starting_money))) - 1

    @classmethod
    def state_size(cls):
        return 4

    @classmethod
    def action_count(cls):
        return 4

    def state(self, current_price, future_price):
        return [
            self.scale_money(self.money),
            self.scale_money(self.valuation(current_price)),
            self.scale_money(self.valuation(future_price)),
            math.copysign(1, self.volume) * math.log(math.fabs(self.volume) + 1)
        ]