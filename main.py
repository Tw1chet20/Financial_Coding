from t212_client import Trading212Client, Executor
from market_data import MarketData
from strategy import Signaller
from state_manager import StateManager
from config import State, FileParams, YAHOO_MAP
import time
import logging
from pathlib import Path
import schedule

logging.basicConfig(
    filename=Path("bot.log"),
    level=logging.INFO
)

client = Trading212Client()
executor = Executor(client)
strategy = Signaller()
state_manager = StateManager()

def y_ticker(t212_ticker: str):
    return YAHOO_MAP[t212_ticker]

def already_pending(ticker: str):

    try:
        orders = client.get_orders()

        for order in orders:
            if (
                order["ticker"] == ticker
                and order["status"] in ["NEW", "PROCESSING"]
            ):
                return True

    except Exception as e:
        logging.exception(e)
        return True

    return False

def update_state(
        state: State,
        new_quantity: float,
        buy: bool,
        close: bool
        ):
    
    if close:

        new_state = {
            "ticker": state["ticker"],
            "target_value": state["target_value"],
            "position_quantity": 0.0,
            "current_position_value": 0.0
        }

        state_manager.save_state(
            new_state
        )
    
    else:

        current_price = client.get_positions()[0]['currentPrice']
        target_value = state['target_value']
        value = state["current_position_value"]
        value += new_quantity*current_price - target_value

        new_state = {
            "ticker": state["ticker"],
            "target_value": target_value,
            "position_quantity": new_quantity,
            "current_position_value": value
        }

        state_manager.save_state(
            new_state
        )

def reconcile_state():

    state_update = False

    state: State = state_manager.load_state()

    actual_quantity = client.get_position_quantity(
        state["ticker"]
    )
    current_price = client.get_positions()[0]['currentPrice']
    actual_position_value = actual_quantity*current_price

    if actual_quantity != state["position_quantity"]:

        logging.warning(
            f"State mismatch on quantity corrected to {actual_quantity}"
        )
        state["position_quantity"] = actual_quantity

        state_update = True

    if actual_position_value != state["current_position_value"]:

        logging.warning(
            f"State mismatch on position value corrected to {actual_position_value}"
        )
        state["current_position_value"] = actual_position_value

        state_update = True

    if state_update:

        state_manager.save_state(state)
        print("\nInitial State Mismatch Resolved")

running = False
bot_running = True

def run_cycle(
        file_params: FileParams,
        just_strat: bool = True,
        risk_up: float = 0.1,
        risk_down: float = -0.1,
        timeout: int = 60,
        delta: float = 0.00001
        ):

    print('\n-------------------------')
    print('Running cycle...')

    global running
    global bot_running

    if running:

        print('Previous cycle running')
        print('\n-------------------------')
        return
    
    running = True

    logging.info("Cycle Started")

    try:

        state: State = state_manager.load_state()
        ticker = state['ticker']
        yf_ticker = y_ticker(ticker)

        current_price = client.get_positions()[0]['currentPrice']
        quantity = state['position_quantity']
        position_value = state["current_position_value"]
        target_value = state["target_value"]
        diff = position_value - target_value

        if diff/target_value >= risk_up:

            logging.info(f"Closing position with profit at {round(diff,2)}")
            print('\nClosing position by taking profit...')
            client.close_position(ticker)

            filled_order = False
            start = time.time()

            while time.time() - start < timeout:

                print(f'\nChecking order fill: {round(time.time() - start,0)}/{timeout} seconds')

                new_quant = client.get_position_quantity(ticker)

                if new_quant == 0:
                    filled_order = True
                    break

                time.sleep(5)

            if filled_order:

                update_state(
                    state,
                    0.0,
                    True,
                    True
                )
                bot_running = False

                logging.info("Position closed")
                print("\nPosition closed")
                print("\nBot ending...")
                print('\n-------------------------')
                return
            
            else:
                
                logging.info("Unable to close position")
                print("\nPosition not closed")
                print("\nContinuing cycle...")
                print('\n-------------------------')
                return
            
        if diff/target_value <= risk_down:

            logging.info(f"Closing position with loss at {round(diff,2)}")
            print('\nClosing position by cutting losses...')
            client.close_position(ticker)

            filled_order = False
            start = time.time()

            while time.time() - start < timeout:

                print(f'\nChecking order fill: {round(time.time() - start,0)}/{timeout} seconds')

                new_quant = client.get_position_quantity(ticker)

                if new_quant == 0:
                    filled_order = True
                    break

                time.sleep(5)

            if filled_order:

                update_state(
                    state,
                    0.0,
                    True,
                    True
                )
                bot_running = False

                logging.info("Position closed")
                print("\nPosition closed")
                print("\nBot ending...")
                print('\n-------------------------')
                return
            
            else:
                
                logging.info("Unable to close position")
                print("\nPosition not closed")
                print("\nContinuing cycle...")
                print('\n-------------------------')
                return

        current_price = client.get_positions()[0]['currentPrice']
        signal = strategy.check(
            current_price=current_price,
            position_quantity=quantity,
            state=state,
            file_params=file_params,
            just_strat=just_strat
            )

        if signal:

            pending = already_pending(ticker)

            if signal['action'] == 'sell' and not pending:

                print('\nExecutor enacted')
                result = executor.execute(signal)

                if result:

                    filled_order = False
                    new_quantity = 0.0
                    order_id = result['id']

                    start = time.time()

                    while time.time() - start < timeout:

                        print(f'\nChecking order fill: {round(time.time() - start,0)}/{timeout} seconds')

                        new_quant = client.get_position_quantity(ticker)

                        if abs(new_quant - quantity) > delta:
                            filled_order = True
                            new_quantity = new_quant
                            break

                        time.sleep(5)

                    if filled_order:

                        logging.info(
                            f"{signal['action']} "
                            f"{signal['quantity']} "
                            f"{ticker} "
                            f"@ {current_price}"
                            )
                        update_state(
                            state,
                            abs(new_quantity),
                            buy=False,
                            close=False
                        )
                        print('\nState Updated')
                        print('\n-------------------------')

                    else:

                        print('\nOrder not filled')
                        print('\n-------------------------')

                        logging.warning(
                            f"Order {order_id} timed out"
                            )

            elif signal['action'] == 'buy' and not pending:

                available_cash = client.get_cash()['free']
        
                if available_cash >= signal['quantity']*current_price:

                    print('\nExecutor enacted')
                    result = executor.execute(signal)

                    if result:

                        filled_order = False
                        new_quantity = 0.0
                        order_id = result['id']

                        start = time.time()

                        while time.time() - start < timeout:

                            print(f'\nChecking order fill: {round(time.time() - start,0)}/{timeout} seconds')

                            new_quant = client.get_position_quantity(ticker)

                            if abs(new_quant - quantity) > delta:
                                filled_order = True
                                new_quantity = new_quant
                                break

                            time.sleep(5)

                        if filled_order:

                            logging.info(
                                f"{signal['action']} "
                                f"{signal['quantity']} "
                                f"{ticker} "
                                f"@ {current_price}"
                                )
                            update_state(
                                state,
                                abs(new_quantity),
                                buy=True,
                                close=False
                            )
                            print('\nState Updated')
                            print('\n-------------------------')

                        else:

                            print('\nOrder not filled')
                            print('\n-------------------------')

                            logging.warning(
                                f"Order {order_id} timed out"
                                )

                else:

                    print('Not Enough Cash')
                    print('\n-------------------------')

        else:
         
            logging.info("No signal found on this cycle")
            print('\nNo Signal')
            print('\n-------------------------')

    except Exception as e:

        logging.exception(e)
    
    finally:

        logging.info("Cycle Finished")

        running = False

file_params: FileParams = {
    "period": "58d",
    "interval": "2m",
    "down_max": 10,
    "up_max": 15,
    "up_retrieval": False
}

print("\nBot started...")
print("-------------------------")
#reconcile_state()
run_cycle(file_params=file_params,just_strat=True)

schedule.every(2).minutes.do(
    run_cycle,
    file_params=file_params,
    just_strat=True
)

while bot_running:

    schedule.run_pending()

    id="k6x7z"
    next_run = schedule.next_run()

    if next_run:
        remaining_seconds = int(
            (next_run - schedule.datetime.datetime.now()).total_seconds()
        )
        minutes, seconds = divmod(remaining_seconds, 60)
        print(
            f"\rNext cycle in: {minutes:02d}:{seconds:02d}",
            end=""
        )

    time.sleep(1)

print('\nBot Stopped')