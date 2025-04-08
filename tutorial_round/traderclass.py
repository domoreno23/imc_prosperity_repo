from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List
import string
import json


class Strategy:
  """
  Abstract Strategy class
  """

  def __init__(self, symbol: str, limit: int):
    self.symbol = symbol
    self.limit = limit
    self.position = 0


  def run(self, state: TradingState) -> list[Order]:
      

      self.orders = []
      for product in state.order_depths: # bids and asks that we can take. Someone is offering to sell or buy
          order_depth: OrderDepth = state.order_depths[product]
          orders: List[Order] = []
          
          match product:
            case 1: 
              self.orders = RainForestResin.run(order_depth, orders, product)
            case 2:
              self.orders = Kelp.run()
              
      return self.orders


class Kelp(Strategy):
  pass


class RainForestResin(Strategy):
  def __init__(self, symbol: str, limit: int):
    super().__init__(symbol=symbol, limit=limit)
  
  def run(self, state: TradingState) -> List[Order]:
    orders=[]
    acceptable_price = 10000
    if self.symbol not in state.order_depths:
      return orders
    else:
      order_depth = state.order_depths[self.symbol]
    
      if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(symbol=self.symbol, best_ask=best_ask, best_ask_amount=-best_ask_amount))
    
      if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(symbol=self.symbol, best_bid=best_bid, best_bid_amount=-best_bid_amount))
      return orders

class Squid_Ink(Strategy):
  pass


class Trader:
    def __init__(self):
    
      # Create each strategy for each product here
      self.strategies = {
        "RAINFOREST_RESIN" : RainForestResin("RAINFOREST_RESIN", 50),
        "KELP": Kelp("KELP", 50)
      }

      self.trader_data = {}  #storing state between rounds

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
				
        if state.traderData:
            try:
              self.trader_data = json.loads(state.traderData)
            except:
              self.trader_data = {}

        for product, strategy in self.strategies.items():
            if product in state.order_depths:
              try:
                  orders = strategy.run(state=state)
                  result[product] = orders
              except:
                print(f"Error executing strategy for {product}")
            
                result[product]
            
            result[product] = orders
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        trader_data_str = json.dumps(self.trader_data)
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, trader_data_str