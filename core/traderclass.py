from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List
import numpy as np
import string
import json
from statistics import NormalDist
from collections import deque


def trade(state:TradingState, symbol: string, is_buy_orders: bool, orders: list[Order], acceptable_price: int):
  
  order_depth = state.order_depths[symbol]  

  if is_buy_orders:
      if len(order_depth.buy_orders) != 0:
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        if int(best_bid) > acceptable_price:       
            orders.append(Order(symbol, best_bid, -best_bid_amount))

  else:
    if len(order_depth.sell_orders) != 0:
          best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
          if int(best_ask) < acceptable_price:
              orders.append(Order(symbol, best_ask, -best_ask_amount))


#trade(state, self.symbol, t/f, orders, acceptable_price)

#take marketprice-theoritcal, which underprices is alot buy that


class VolcanicVoucher:
    def __init__(self, symbol: str, strike_price: float, expiry_days: int, limit: int):
        self.symbol = symbol
        self.strike_price = strike_price
        self.expiry_days = expiry_days
        self.limit = limit
        self.vol_window = 20
        self.vol_history = []

    def black_scholes_call(self, S, K, T, sigma, r=0.0):
        """Black-Scholes European call option price"""
        if sigma == 0 or T == 0:
            return max(0, S - K)
        N = NormalDist().cdf
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * N(d1) - K * np.exp(-r * T) * N(d2)
        

    def get_mid_price(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None

    def estimate_volatility(self):
        returns = np.diff(np.log(self.vol_history))
        return np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

    def run(self, state: TradingState) -> List[Order]:
        orders = []

        if ("VOLCANIC_ROCK" not in state.order_depths) or (len(state.order_depths["VOLCANIC_ROCK"] == 0) and
            len(state.order_depth["VOLCANIC_ROCK"]) == 0):
          return
        
        rock_depth = state.order_depths["VOLCANIC_ROCK"]
        voucher_depth = state.order_depths[self.symbol]
           
        S = self.get_mid_price(rock_depth)
        V = self.get_mid_price(voucher_depth)

        if S is None or V is None:
            return []

        # Update volatility estimate
        self.vol_history.append(S)
        if len(self.vol_history) > self.vol_window:
            self.vol_history.pop(0)

        sigma = self.estimate_volatility()

        # Time to expiration in years (e.g., 7 days = 7/252)
        T = self.expiry_days / 252.0

        # Compute theoretical value
        theoretical_value = self.black_scholes_call(S, self.strike_price, T, sigma)

        # Get market quotes
        best_ask, ask_vol = list(voucher_depth.sell_orders.items())[0] if voucher_depth.sell_orders else (None, None)
        best_bid, bid_vol = list(voucher_depth.buy_orders.items())[0] if voucher_depth.buy_orders else (None, None)

        # Buy undervalued options
        if best_ask is not None and best_ask < theoretical_value:
            volume = min(-ask_vol, self.limit)
            orders.append(Order(self.symbol, best_ask, volume))

        # Sell overvalued options
        if best_bid is not None and best_bid > theoretical_value:
            volume = min(bid_vol, self.limit)
            orders.append(Order(self.symbol, best_bid, -volume))

        print(f"[{self.symbol}] Fair: {theoretical_value:.2f}, Ask: {best_ask}, Bid: {best_bid}, Volatility: {sigma:.4f}")

        return orders



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
      if any(symbol not in state.order_depths for symbol in ["JAMS", "CROISSANTS", "DJEMBE"]):
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
      
      trade(state, self.symbol, True, orders, acceptable_price)
      trade(state, self.symbol, False, orders, acceptable_price)  
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
      if any(symbol not in state.order_depths for symbol in ["JAMS", "CROISSANTS", "DJEMBE"]):
        expectedValue1 = self.last_price
      else:
        jams= self.products["JAMS"].get_mid_price(state.order_depths["JAMS"])
        croissants = self.products["CROISSANTS"].get_mid_price(state.order_depths["CROISSANTS"])
        expectedValue1 = croissants*4 + jams*2
      orders=[]
     

      trade(state, self.symbol, True, orders, expectedValue1)
      trade(state, self.symbol, False, orders, expectedValue1)
      
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
            volume = min(-best_ask_amount, self.limit)
            orders.append(Order(self.symbol, best_ask, volume))
      
      if len(order_depth.buy_orders) != 0:
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        if int(best_bid) > expectedValue2:       
            volume = min(best_bid_amount, self.limit)
            orders.append(Order(self.symbol, best_bid, volume))
      
      

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
            volume = min(-best_ask_amount, self.limit)
            orders.append(Order(self.symbol, best_ask, volume))
        
      if len(order_depth.buy_orders) != 0:
          best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
          if int(best_bid) > acceptable_price:       
              volume = min(best_ask_amount, self.limit)
              orders.append(Order(self.symbol, best_bid, volume))

    
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
            volume = min(-best_ask_amount, self.limit)
            orders.append(Order(self.symbol, best_ask, volume))
      
    if len(order_depth.buy_orders) != 0:
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        if int(best_bid) > acceptable_price:
            volume = min(best_ask_amount, self.limit)
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

    trade(state, self.symbol, True, orders, acceptable_price)
    trade(state, self.symbol, False, orders, acceptable_price)
  
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
          orders.append(Order(self.symbol, 9992, 25))
          
    
    if len(order_depth.buy_orders) != 0:
      best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
      if int(best_bid) > acceptable_price:       
          orders.append(Order(self.symbol, best_bid, -best_bid_amount))
          orders.append(Order(self.symbol, 10008, -25))
          
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
        "PICNIC_BASKET2": Basket2("PICNIC_BASKET2", 100), 
        "VOLCANIC_ROCK_VOUCHER_9500": VolcanicVoucher("VOLCANIC_ROCK_VOUCHER_9500", 9500, expiry_days=7, limit=200),
        "VOLCANIC_ROCK_VOUCHER_9750": VolcanicVoucher("VOLCANIC_ROCK_VOUCHER_9750", 9750, expiry_days=7, limit=200),
        "VOLCANIC_ROCK_VOUCHER_10000": VolcanicVoucher("VOLCANIC_ROCK_VOUCHER_10000", 10000, expiry_days=7, limit=200),
        "VOLCANIC_ROCK_VOUCHER_10250": VolcanicVoucher("VOLCANIC_ROCK_VOUCHER_10250", 10250, expiry_days=7, limit=200),
        "VOLCANIC_ROCK_VOUCHER_10500": VolcanicVoucher("VOLCANIC_ROCK_VOUCHER_10500", 10500, expiry_days=7, limit=200),
      }


      for product in self.strategies:
            if hasattr(self.strategies[product], "products"):
                self.strategies[product].products = self.strategies

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
