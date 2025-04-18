from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List
import numpy as np
import string
import json
from statistics import NormalDist
from collections import deque
import pandas as pd
######################



#Global trade function used by any class
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


class MagnificientMacroons:
    def __init__(self, symbol: str, limit: int):
        self.symbol = symbol
        self.limit = limit
        self.data = []
        self.sunlight_window = deque(maxlen=100) #Last 100 sunlight scores
        self.csi = 55 #critical sunlight index

    def run(self, state: TradingState):
        orders = []

        # Get order book
        depth = state.order_depths[self.symbol]
        best_bid, best_bid_amount = list(depth.buy_orders.items())[0]
        best_ask, best_ask_amount = list(depth.sell_orders.items())[0]
        macaron_mid_price = (best_bid + best_ask) / 2

        # Get features
        obs = state.observations.conversionObservations[self.symbol]
        sunlight = obs.sunlightIndex
        sugar = obs.sugarPrice
        transport = obs.transportFees
        import_tariff = obs.importTariff
        export_tariff = obs.exportTariff
        
        #Sunlight history
        self.sunlight_window.append(sunlight)
        low_sunlight_count = sum(s < self.csi for s in self.sunlight_window)
        low_sunlight_ratio = low_sunlight_count / len(self.sunlight_window)
        
        #Used ML model to compute the coefficients and intercept locally
        a, b, c, d, e = -0.74460938, 12.62199989,  36.16015896, -31.11214568,  26.93065467
        base_fair_price = -2286.778291436136 + (a * sunlight) + (b * sugar) + (c * transport) + (d * import_tariff) + (e * export_tariff)
        
        if low_sunlight_ratio >= 0.7:
            # If more than 70% of last 10 ticks are under CSI → panic pricing expected
            panic_multiplier = 1.15  # or try 1.15, depending on historical uplift
            fair_price = base_fair_price * panic_multiplier
        else:
            fair_price = base_fair_price
        
        # === Position logic ===
        position = state.position.get(self.symbol, 0)
        max_buy = self.limit - position
        max_sell = self.limit + position

        

        if fair_price > best_ask:
            volume = min(max_buy, best_ask_amount)
            if volume > 0:
                orders.append(Order(self.symbol, best_ask, -volume))

        if fair_price < best_bid:
            volume = min(max_sell, best_bid_amount)
            if volume > 0:
                orders.append(Order(self.symbol, best_bid, volume))

        return orders






class Croissants():
    def __init__(self, symbol: str, limit: int):
      self.symbol = symbol
      self.limit = limit
      self.last_basket1 = 0

    def get_mid_price(self, state: TradingState):
        order_depth = state.order_depths["CROISSANTS"]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        jams_mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        return jams_mid_price
    
    def run(self, state: TradingState):

      
      orders = []
      basket1Price = Basket1.basket1Price(self, state)
      orders = []
      order_depth = state.order_depths["CROISSANTS"]
      best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
      best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
      expectedValue = .1 * basket1Price
      if best_bid > expectedValue:
         orders.append(Order(self.symbol, best_bid, -10))
      if best_ask < expectedValue:
         orders.append(Order(self.symbol, best_ask, 10))
    
      return orders


class Jams():
    def __init__(self, symbol: str, limit: int):
      self.symbol = symbol
      self.limit = limit
      self.last_croissant = 0
      self.last_djembe = 0
      
    
    def get_mid_price(self, state):
        order_depth = state.order_depths["JAMS"]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        jams_mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        return jams_mid_price
    
    def run(self, state:TradingState):
      basket1Price = Basket1.basket1Price(self, state)
      orders = []
      order_depth = state.order_depths["JAMS"]
      best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
      best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
      expectedValue = .1 * basket1Price
      if best_bid > expectedValue:
         orders.append(Order(self.symbol, best_bid, -10))
      if best_ask < expectedValue:
         orders.append(Order(self.symbol, best_ask, 10))
      
      return orders


#trade(state, self.symbol, t/f, orders, acceptable_price)

#take marketprice-theoritcal, which underprices is alot buy that

#SUM UP POSITIONS AND TAKE THE OPPOSTITE OF THAT (BUY/SHORT) AND DO THAT

#Carlos will solve this

class VolcanicRock():
  def __init__(self, symbol: str, limit: int):
    self.symbol = symbol
    self.limit = limit
    self.vol_history = []
    self.vol_window = 20
    self.vouchers = {
       "VOLCANIC_ROCK_VOUCHER_9500": 9500,
        "VOLCANIC_ROCK_VOUCHER_9750": 9750, 
        "VOLCANIC_ROCK_VOUCHER_10000": 10000,
        "VOLCANIC_ROCK_VOUCHER_10250": 10250
    }
    self.min_volatility = 0.20
    

  def get_mid_price(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None
  

  def calculate_delta(self, S: float, K: float, T: float, sigma: float):
        try:
            if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
                return 0.0
            sigma = max(sigma, self.min_volatility)
            
            d1 = (np.log(S/K) + (0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return NormalDist().cdf(d1)
        except:
            return 0.0  # Fallback if calculation fails

    


  def estimate_volatility(self):
        if len(self.vol_history) < 2:
          return 0.20
        returns = np.diff(np.log(self.vol_history))
        return np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
    
  def run(self, state: TradingState) -> List[Order]:
    #Potentially, everytime a voucher is run, i run this class, and i check the position of this classs

    orders = []

    rock_mid_price = self.get_mid_price(state.order_depths[self.symbol])

    self.vol_history.append(rock_mid_price)
    if len(self.vol_history) > self.vol_window:
            self.vol_history.pop(0)

    sigma = self.estimate_volatility()
    T = 5
    sum_delta = 0

    rock_depth = state.order_depths[self.symbol]
    for voucher, strike_price in self.vouchers.items():
      if voucher in state.order_depths:
        delta = self.calculate_delta(rock_mid_price, strike_price, T, sigma )
        sum_delta += state.position.get(voucher, 0) * delta

    req_rock_position = -round(sum_delta)
    current_rock_position = state.position.get(self.symbol, 0)

    position_diff = req_rock_position - current_rock_position

    if position_diff > 0:
            # Need to buy rock
            best_ask, best_ask_vol = next(iter(rock_depth.sell_orders.items()), (None, None))
            if best_ask:
                volume = min(position_diff, best_ask_vol, self.limit - current_rock_position)
                orders.append(Order(self.symbol, best_ask, volume))
    elif position_diff < 0:
            # Need to sell rock
            best_bid, best_bid_vol = next(iter(rock_depth.buy_orders.items()), (None, None))
            if best_bid:
                volume = max(position_diff, -best_bid_vol, -self.limit - current_rock_position)
                orders.append(Order(self.symbol, best_bid, volume))

    return orders

         
    
    
    
 
  




class VolcanicVoucher():
    def __init__(self, symbol: str, strike_price: float, expiry_days: int, limit: int):
        self.symbol = symbol
        self.strike_price = strike_price
        self.expiry_days = expiry_days
        self.limit = limit
        self.vol_window = 20
        self.vol_history = []

    def get_mid_price(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None
    
    def black_scholes_call(self, S, K, T, sigma, r=0.0):
        """Black-Scholes European call option price"""
        if sigma == 0 or T == 0:
            return max(0, S - K)
        N = NormalDist().cdf
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * N(d1) - K * np.exp(-r * T) * N(d2)
      
    #realized volatility
    def estimate_volatility(self):
        returns = np.diff(np.log(self.vol_history))
        return np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

    def run(self, state: TradingState) -> List[Order]:
        orders = []

        if ("VOLCANIC_ROCK" not in state.order_depths or len(state.order_depths["VOLCANIC_ROCK"].buy_orders) == 0 or 
            len(state.order_depths["VOLCANIC_ROCK"].sell_orders) == 0):
          return
        
        rock_depth = state.order_depths["VOLCANIC_ROCK"]
        voucher_depth = state.order_depths[self.symbol]
          
        rock_mid_price = self.get_mid_price(rock_depth)
        voucher_mid_price = self.get_mid_price(voucher_depth)

        #print(rock_mid_price)

        if rock_mid_price is None or voucher_mid_price is None:
            return []

        # Update volatility estimate
        self.vol_history.append(rock_mid_price)
        if len(self.vol_history) > self.vol_window:
            self.vol_history.pop(0)

        sigma = self.estimate_volatility()

        T = self.expiry_days 

        # Compute theoretical value
        theoretical_value = self.black_scholes_call(rock_mid_price, self.strike_price, T, sigma)

        # Get market quotes
        best_ask, ask_vol = list(voucher_depth.sell_orders.items())[0] if voucher_depth.sell_orders else (None, None)
        best_bid, bid_vol = list(voucher_depth.buy_orders.items())[0] if voucher_depth.buy_orders else (None, None)

        best_rock_ask, ask_rock_vol = list(rock_depth.sell_orders.items())[0] if voucher_depth.sell_orders else (None, None)
        best_rock_bid, ask_rock_vol = list(rock_depth.buy_orders.items())[0] if voucher_depth.sell_orders else (None, None)

        # Buy undervalued options
        if best_ask is not None and best_ask < theoretical_value:
            #volume = min(ask_vol, self.limit)
            orders.append(Order(self.symbol, best_ask, -15))
            #orders.append(Order("VOLCANIC_ROCK", best_rock_ask, 15))
          

        # Sell overvalued options
        if best_bid is not None and best_bid > theoretical_value:
            #volume = min(bid_vol, self.limit)
            orders.append(Order(self.symbol, best_bid, -15))
            #orders.append(Order("VOLCANIC_ROCK", best_rock_bid, 15))

        #print(f"[{self.symbol}] Fair: {theoretical_value:.2f}, Ask: {best_ask}, Bid: {best_bid}, Volatility: {sigma:.4f}")

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
            orders.append(Order(self.symbol, best_bid, volume))
    
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
    window = 101
    position = state.position.get(self.symbol, 0)
    z_value, best_ask_value, best_bid_value = self.calculate_z_score(order_depth, window_size=window)
    
    #This threshold can also be used for tuning the algorithm
    z_threshold = 3 
    
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
        #"CROISSANTS": Croissants("CROISSANTS", 250),
        "JAMS": Jams("JAMS", 350),
        "DJEMBE": Djembe("DJEMBE", 60),
        "PICNIC_BASKET2": Basket2("PICNIC_BASKET2", 100), 
        "VOLCANIC_ROCK_VOUCHER_9500": VolcanicVoucher("VOLCANIC_ROCK_VOUCHER_9500", 9500, expiry_days=5, limit=200),
        "VOLCANIC_ROCK_VOUCHER_9750": VolcanicVoucher("VOLCANIC_ROCK_VOUCHER_9750", 9750, expiry_days=5, limit=200),
        "VOLCANIC_ROCK_VOUCHER_10000": VolcanicVoucher("VOLCANIC_ROCK_VOUCHER_10000", 10000, expiry_days=5, limit=200),
        "VOLCANIC_ROCK_VOUCHER_10250": VolcanicVoucher("VOLCANIC_ROCK_VOUCHER_10250", 10250, expiry_days=5, limit=200),
        "VOLCANIC_ROCK_VOUCHER_10500": VolcanicVoucher("VOLCANIC_ROCK_VOUCHER_10500", 10500, expiry_days=5, limit=200),
        "VOLCANIC_ROCK": VolcanicRock("VOLCANIC_ROCK", limit = 400),
        "MAGNIFICENT_MACARONS": MagnificientMacroons("MAGNIFICENT_MACARONS",75)
      }


      for product in self.strategies:
            if hasattr(self.strategies[product], "products"):
                self.strategies[product].products = self.strategies
            

      

      self.trader_data = {}  #storing state between rounds

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        #For each product, check if product is in orderDepth. If so run strategy
        for product, strategy in self.strategies.items():
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
