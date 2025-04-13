from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List
import numpy as np
import string
import json
from collections import deque


class Basket1():
    def __init__(self, symbol: str, limit: int):
      self.symbol = symbol
      self.limit = limit
      self.last_jam = 0
      self.last_croissant = 0
      self.last = 0
      self.products = None #Stores the references to each product here

    
    def basket1Price(self, state: TradingState):
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        basket_mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        return basket_mid_price
    
    def run(self, state:TradingState):
      expectedValue = 0
      #All prices for jams
      if any(symbol not in state.order_depths for symbol in ["JAMS, CROISSANTS, DJEMBE"]):
        expectedValue = self.last
      else:
        jams= self.products["JAMS"].get_mid_price(state.order_depths["JAMS"])
        croissants = self.products["CROISSANTS"].get_mid_price(state.order_depths["CROISSANTS"])
        djembe = self.products["DJEMBE"].get_mid_price(state.order_depths["DJEMBE"])
        

        expectedValue = croissants*6 + jams*3 + djembe #jamsExpectedValue = basket - 3croissant - djembe
        ##Basket1 Position Size: 20,40,60,80,100
       ##Croissant Pos size:
       ##jams Position size: 3,6,9,12,15
       #djembe position size: 1,2,3,4,5

      

      orders = []
      acceptable_price = expectedValue
      order_depth = state.order_depths[self.symbol]   
      if len(order_depth.sell_orders) != 0:
        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
       
        if int(best_ask) < acceptable_price:
            orders.append(Order(self.symbol, best_ask, -best_ask_amount))
      
      if len(order_depth.buy_orders) != 0:
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        if int(best_bid) > acceptable_price:       
            orders.append(Order(self.symbol, best_bid, -best_bid_amount))
            
      self.last = expectedValue
      return orders



class Basket2():
    def __init__(self, symbol: str, limit: int):
      self.symbol = symbol
      self.limit = limit
      self.last_price = 0
      self.products = None #Stores the references to each product here
      
   
    def basket2Price(self, state: TradingState):
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        basket_mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        return basket_mid_price

    def run(self, state: TradingState):
      expectedValue1 = 0
      if any(symbol not in state.order_depths for symbol in ["JAMS, CROISSANTS, DJEMBE"]):
        expectedValue1 = self.last_price
      else:
        jams= self.products["JAMS"].get_mid_price(state.order_depths["JAMS"])
        croissants = self.products["CROISSANTS"].get_mid_price(state.order_depths["CROISSANTS"])
        expectedValue1 = croissants*4 + jams*2
      orders=[]
      order_depth = state.order_depths["PICNIC_BASKET2"]

      if len(order_depth.sell_orders) != 0:
        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        if int(best_ask) < expectedValue1:
            orders.append(Order(self.symbol, best_ask, -best_ask_amount))
      
      if len(order_depth.buy_orders) != 0:
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        if int(best_bid) > expectedValue1:       
            orders.append(Order(self.symbol, best_bid, -best_bid_amount))
      self.last_price = expectedValue1
      return orders

class Jams():
    def __init__(self, symbol: str, limit: int):
      self.symbol = symbol
      self.limit = limit
      self.last_croissant = 0
      self.last_djembe = 0
      self.products = None
    
    def get_mid_price(self, order_depth: OrderDepth):
        
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        jams_mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        return jams_mid_price
    
    
    
    def run(self, state:TradingState):
      croissants = (self.last_croissant if "CROISSANTS" not in state.order_depths 
                    else self.products["CROISSANTS"].get_mid_price(state.order_depths["CROISSANTS"]))
      djembe = (self.last_djembe if "DJEMBE" not in state.order_depths 
                else self.products["DJEMBE"].get_mid_price(state.order_depths["DJEMBE"]))
      
      expectedValue1 = (-6*croissants + -djembe+ self.products["PICNIC_BASKET1"].basket1Price(state))/3
      expectedValue2 = (self.products["PICNIC_BASKET2"].basket2Price(state) - croissants*4)/2
      
      orders = []#Submitted Orders

      #print("Expected Value: ", expectedValue1, "\n")
      
      
      order_depth = state.order_depths[self.symbol] 
      #For expected Value 1 from first basket  
      if len(order_depth.sell_orders) != 0:
        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        if int(best_ask) < expectedValue1:
            orders.append(Order(self.symbol, best_ask, -best_ask_amount))
      
      if len(order_depth.buy_orders) != 0:
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        if int(best_bid) > expectedValue1:       
            orders.append(Order(self.symbol, best_bid, -best_bid_amount))
      
      

      self.last_croissant = croissants
      self.last_djembe = djembe
      return orders



class Croissants():
    def __init__(self, symbol: str, limit: int):
      self.symbol = symbol
      self.limit = limit
      self.last_basket1 = 0
      self.products = None

    def get_mid_price(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        croissants_mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        return croissants_mid_price

    def run(self, state: TradingState):
      
      order_depth = state.order_depths[self.symbol]
      expected_value1 = (self.last_basket1 if "PICNIC_BASKET1" not in state.order_depths 
                         else .1 * self.products["PICNIC_BASKET1"].basket1Price(state))
      
      orders = []
      acceptable_price = expected_value1
      if len(order_depth.sell_orders) != 0:
        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        if int(best_ask) < acceptable_price:
            orders.append(Order(self.symbol, best_ask, -best_ask_amount))
        
      if len(order_depth.buy_orders) != 0:
          best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
          if int(best_bid) > acceptable_price:       
              orders.append(Order(self.symbol, best_bid, -best_bid_amount))

     
      self.last_basket1 = expected_value1
      return orders




class Djembe():
  def __init__(self,symbol:str, limit: int):
      self.symbol = symbol
      self.limit = limit
      self.last_croissant = 0
      self.last_djembe = 0
      self.products = None

  def get_mid_price(self, order_depth: OrderDepth):
    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    croissants_mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None
    return croissants_mid_price
  

  def run(self, state: TradingState):
    croissants = self.last_croissant if "CROISSANTS" not in state.order_depths else self.products["CROISSANTS"].get_mid_price(state.order_depths["CROISSANTS"])
    djembe = self.last_djembe if "DJEMBE" not in state.order_depths else self.products["DJEMBE"].get_mid_price(state.order_depths["DJEMBE"])
    expectedValue = .1 * Basket1.basket1Price(state)
      
    orders = []
    acceptable_price = expectedValue
    order_depth = state.order_depths[self.symbol]   
    if len(order_depth.sell_orders) != 0:
        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        if int(best_ask) < acceptable_price:
            orders.append(Order(self.symbol, best_ask, -best_ask_amount))
      
    if len(order_depth.buy_orders) != 0:
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        if int(best_bid) > acceptable_price:       
            orders.append(Order(self.symbol, best_bid, -best_bid_amount))
    
    self.last_croissant = croissants
    self.last_djembe = djembe
    
    
    return orders

      
    












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
                                                    ###orderJams = state.order_depths["Jams"]
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
    self.window: Dict[str, List[int]] = {}  # Keeps a rolling list of prices per product

  def calculate_z_score(self, order_depth: OrderDepth, window_size: int):
    z_score = 0
    #Getting the best ask and sell order
    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    current_mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None
    
    #Used to get a dictionary with the last *window_size value* of Squid Ink
    last_windowsize_prices = self.window.setdefault(self.symbol, [])
    
    # Update price history and make sure the last *window_size value* are in the list
    if current_mid_price:
        last_windowsize_prices.append(current_mid_price)
        if len(last_windowsize_prices) > window_size:  
            last_windowsize_prices.pop(0)
    
    #Calculating the z score
    if len(last_windowsize_prices) >= window_size:
      sma = np.mean(last_windowsize_prices)
      std = np.std(last_windowsize_prices)
      z_score = ((current_mid_price - sma)/std)
      
    return z_score, best_ask, best_bid

  def run(self, state: TradingState) -> List[Order]:
    order_depth: OrderDepth = state.order_depths[self.symbol]
    orders: List[Order] = []
    window = 100
    position = state.position.get(self.symbol, 0)
    z_value, best_ask_value, best_bid_value = self.calculate_z_score(order_depth, window_size=window)
    
    #This threshold can also be used for tuning the algorithm
    z_threshold = 4 
    
    #Mean Reversion Logic
    # If price is too low → Buy expecting rebound
    if z_value < -z_threshold and best_ask_value:
        volume = min(5, window - position)
        orders.append(Order(self.symbol, best_ask_value, volume))
    
    # If price is too high → Sell expecting drop
    if z_value > z_threshold and best_bid_value:
      volume = min(5, position + window)
      orders.append(Order(self.symbol, best_bid_value, -volume))
    
    return orders


class Trader:
    def __init__(self):
    
      # Create each strategy for each product here
      self.strategies = {
        "RAINFOREST_RESIN" : RainForestResin("RAINFOREST_RESIN", 50),
        "KELP": Kelp("KELP", 50),
        "SQUID_INK": SquidInk("SQUID_INK", 50), 
        "PICNIC_BASKET1": Basket1("PICNIC_BASKET1", 60),
        "CROISSANTS": Croissants("CROISSANTS", 250),
        "JAMS": Jams("JAMS", 350),
        "DJEMBE": Djembe("DJEMBE", 60),
        "PICNIC_BASKET2": Basket2("PICNIC_BASKET2", 100)
      }


      self.strategies["PICNIC_BASKET1"].products = self.strategies
      self.strategies["PICNIC_BASKET2"].products = self.strategies
      self.strategies["JAMS"].products = self.strategies
      self.strategies["CROISSANTS"].products = self.strategies
      self.strategies["DJEMBE"].products = self.strategies

      self.trader_data = {}  #storing state between rounds

    def run(self, state: TradingState):
        print(state.position)
        result = {}

        #For each product, check if product is in orderDepth. If so run strategy
        for product, strategy in self.strategies.items():
            if product in ["JAMS", "CROISSANTS"]:
               continue
            if product in state.order_depths:
              try:
                  orders = strategy.run(state=state)
                  result[product] = orders
              except Exception as e:
                print(f"Error executing strategy for {product}: {str(e)}")
            
                
       
  
        trader_data_str = json.dumps(self.trader_data)
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, trader_data_str