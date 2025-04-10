from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List
import string
import json
from collections import deque

class Kelp():
  def __init__(self, symbol, limit):
    self.symbol = symbol
    self.limit = limit

  ##Calculating the acceptable price
  def calculate_value(self, order_depth: OrderDepth, depth: int):
     if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
      buy_orders = order_depth.buy_orders
      sell_orders = order_depth.sell_orders
      
      #Sort the top n volumes in the each side
      top_buys = sorted(buy_orders.items(), key=lambda x: -x[0])[:depth]
      top_sells = sorted(sell_orders.items(), key=lambda x: x[0])[:depth]

      total_volume = 0 
      weighted_sum = 0

      #Calculate the weighted average
      for price, volume in top_buys + top_sells:
          total_volume += abs(volume)
          weighted_sum += price * abs(volume)

      return weighted_sum / total_volume if total_volume != 0 else None
    
  def run(self, state: TradingState):
    orders=[]
    
    order_depth = state.order_depths[self.symbol]
    acceptable_price = self.calculate_value(order_depth, 5)
           
    if len(order_depth.sell_orders) != 0:
      best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
      if int(best_ask) < acceptable_price:
          orders.append(Order(self.symbol, best_ask, -best_ask_amount))
    
    if len(order_depth.buy_orders) != 0:
      best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
      if int(best_bid) > acceptable_price:       
          orders.append(Order(self.symbol, best_bid, -best_bid_amount))
  
    return orders
        
class RainForestResin():
  def __init__(self, symbol: str, limit: int):
    self.symbol = symbol
    self.limit = limit
  
  def run(self, state: TradingState) -> List[Order]:
    orders=[]
    acceptable_price = 10000
    order_depth = state.order_depths[self.symbol]
    if len(order_depth.sell_orders) != 0:
      best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
      if int(best_ask) < acceptable_price:
          orders.append(Order(self.symbol, best_ask, -best_ask_amount))
    
    if len(order_depth.buy_orders) != 0:
      best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
      if int(best_bid) > acceptable_price:       
          orders.append(Order(self.symbol, best_bid, -best_bid_amount))
    return orders



class SquidInk():
  def __init__(self, symbol: str, limit: int):
    self.symbol = symbol
    self.limit = limit
    self.window = deque

  def calculate_value(self, order_depth: OrderDepth, depth: int):
    if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
      buy_orders = order_depth.buy_orders
      sell_orders = order_depth.sell_orders
      
      #Sort the top n volumes in the each side
      top_buys = sorted(buy_orders.items(), key=lambda x: -x[0])[:depth]
      top_sells = sorted(sell_orders.items(), key=lambda x: x[0])[:depth]

      total_volume = 0 
      weighted_sum = 0

      #Calculate the weighted average
      for price, volume in top_buys + top_sells:
          total_volume += abs(volume)
          weighted_sum += price * abs(volume)

      average = weighted_sum / total_volume if total_volume != 0 else None

      if average is not None:
        if len(self.window) >= 60:
          self.window.popleft(0)
        self.window.append(average)

      return sum(self.window) / len(self.window) if len(self.window) != 0 else None

  def run(self, state: TradingState) -> List[Order]:
    order_depth: OrderDepth = state.order_depths[self.symbol]
    orders: List[Order] = []
    acceptable_price = self.calculate_value(order_depth, 5)

    if len(order_depth.sell_orders) != 0:
      best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
      if int(best_ask) < acceptable_price:
          orders.append(Order(self.symbol, best_ask, -best_ask_amount))
    
    if len(order_depth.buy_orders) != 0:
      best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
      if int(best_bid) > acceptable_price:       
          orders.append(Order(self.symbol, best_bid, -best_bid_amount))

    return orders
  


class Trader:
    def __init__(self):
    
      # Create each strategy for each product here
      self.strategies = {
        "RAINFOREST_RESIN" : RainForestResin("RAINFOREST_RESIN", 50),
        "KELP": Kelp("KELP", 50),
        "SQUID_INK": SquidInk("SQUID_INK", 50)
      }

      self.trader_data = {}  #storing state between rounds

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}

        #For each product, check if product is in orderDepth. If so run strategy
        for product, strategy in self.strategies.items():
            if product in state.order_depths:
              try:
                  orders = strategy.run(state=state)
                  result[product] = orders
              except Exception as e:
                print(f"Error executing strategy for {product}: {str(e)}")
            
                result[product]
            
            result[product] = orders
  
        trader_data_str = json.dumps(self.trader_data)
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, trader_data_str