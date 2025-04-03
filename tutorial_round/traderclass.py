from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List
import string
import json

class Trader:

    position_limit = {
        "RAINFOREST_RESIN" : 50,
        "KELP" : 50
    }
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

				# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths: # bids and asks that we can take. Someone is offering to sell or buy
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # TODO: calculate this value maybe make a method for this les go
            acceptable_price = 10000# Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price)) # a fair price, what it should revert to
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData
      
      
      ## Imm a delete this part

      
