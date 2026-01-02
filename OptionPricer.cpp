// clang++ -std=c++20 OptionPricer.cpp -o op_pricer

#include <iostream>
#include <random>
#include <vector>
#include <numeric>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;

double normalCDF(double x);
double normalPDF(double x);

enum class OptionKind {
    VanillaCall,
    VanillaPut,
    Combo,
    Exotic
};

class Option {
    public:
        virtual double payoff(const double &stock_price) const = 0;
        virtual vector<double> strikes() const = 0;
        virtual OptionKind kind() const = 0;
        virtual vector<double> weights() const = 0;
        virtual vector<bool> isCall() const = 0;
        virtual ~Option() = default;
};

class MarketData {
    public:
        explicit MarketData(double S, double sigma, double r, double t) : S_(S), sigma_(sigma), r_(r), t_(t) {};

        double spot_price() const { return S_; }
        double volatility() const { return sigma_; }
        double risk_free_rate() const { return r_; }
        double maturity() const { return t_; }
    private:
        const double S_;
        const double sigma_;
        const double r_;
        const double t_;
};

class OptionPricer {
    public:
        virtual double price(const Option& option, const MarketData& marketData) const = 0;
        virtual double delta(const Option& option, const MarketData& marketData) const = 0;
        virtual double gamma(const Option& option, const MarketData& marketData) const = 0;
        virtual double rho(const Option& option, const MarketData& marketData) const = 0;
        virtual double vega(const Option& option, const MarketData& marketData) const = 0;
        
        virtual ~OptionPricer() = default;
};

class EuropeanCall : public Option {
    public:
        explicit EuropeanCall(double strike_price) : strike_price_(strike_price) {};
        double payoff(const double& stock_price) const override {
            return std::max(stock_price - strike_price_, 0.0);
        };
        vector<double> strikes() const override { return {strike_price_}; }
        OptionKind kind() const override { return OptionKind::VanillaCall; }
        vector<double> weights() const override { return {1.0}; }
        vector<bool> isCall() const override { return {true}; }
    private:
        const double strike_price_;
};

class EuropeanPut : public Option {
    public:
        explicit EuropeanPut(double strike_price) : strike_price_(strike_price) {};
        double payoff(const double& stock_price) const override {
            return std::max(strike_price_ - stock_price, 0.0);
        };
        vector<double> strikes() const override { return {strike_price_}; }
        OptionKind kind() const override { return OptionKind::VanillaPut; }
        vector<double> weights() const override { return {1.0}; }
        vector<bool> isCall() const override { return {false}; }
    private:
        const double strike_price_;
};

class Straddle : public Option {
    public:
        explicit Straddle(double strike_price) : strike_price_(strike_price) {};
        double payoff(const double& stock_price) const override {
            return std::abs(stock_price - strike_price_);
        };
        vector<double> strikes() const override { return {strike_price_,strike_price_}; }
        OptionKind kind() const override { return OptionKind::Combo; }
        vector<double> weights() const override { return {1.0,1.0}; }
        vector<bool> isCall() const override { return {true,false}; }
    private:
        const double strike_price_;
};

class Butterfly : public Option {
    public:
        explicit Butterfly(double K1, double K2, double K3) : K1_(K1), K2_(K2), K3_(K3) {};
        double payoff(const double& stock_price) const override {
            return std::max(stock_price - K1_,0.0) - 2*std::max(stock_price - K2_,0.0) + std::max(stock_price - K3_,0.0);
        };
        vector<double> strikes() const override { return {K1_,K2_,K3_}; }
        OptionKind kind() const override { return OptionKind::Combo; }
        vector<double> weights() const override { return {1.0,-2.0,1.0}; }
        vector<bool> isCall() const override { return {true,true,true}; }
    private:
        const double K1_, K2_, K3_;
};

class BS_Pricer : public OptionPricer {
    public:
        double price(const Option& option, const MarketData& marketData) const override {
            double S = marketData.spot_price();
            double sig = marketData.volatility();
            double r = marketData.risk_free_rate();
            double t = marketData.maturity();
            OptionKind kind = option.kind();
            vector<double> Ks = option.strikes();
            vector<double> weights = option.weights();
            vector<bool> isCall = option.isCall();

            if (t == 0) {
                switch (kind) {
                    case OptionKind::VanillaCall: {
                        return std::max(S - Ks[0],0.0);
                    }
                    case OptionKind::VanillaPut: {
                        return std::max(Ks[0] - S,0.0);
                    }
                    case OptionKind::Combo: {
                        if (Ks.size() == weights.size() && Ks.size() == isCall.size()) {
                            double sum = 0.0;
                            for (int i = 0; i < Ks.size(); ++i) {
                                if (isCall[i]) {
                                    sum += std::max(S - Ks[i],0.0);
                                } else {
                                    sum += std::max(Ks[i] - S,0.0);
                                };
                            };
                            return sum;
                        } else {
                            throw runtime_error("Class definitions not valid");
                        };
                    }
                    case OptionKind::Exotic: {
                        throw runtime_error("Black-Scholes doesn't work for this option type.");
                        break;
                    }
                };
            } else if (t > 0) {
                switch (kind) {
                    case OptionKind::VanillaCall: {
                        const double d1 = (std::log(S/Ks[0]) + (r + sig*sig/2)*t)/(sig*std::sqrt(t));
                        const double d2 = d1 - sig*std::sqrt(t);
                        return S*normalCDF(d1) - Ks[0]*std::exp(-r*t)*normalCDF(d2);
                    }
                    case OptionKind::VanillaPut: {
                        const double d1 = (std::log(S/Ks[0]) + (r + sig*sig/2)*t)/(sig*std::sqrt(t));
                        const double d2 = d1 - sig*std::sqrt(t);
                        return Ks[0]*std::exp(-r*t)*normalCDF(-d2) - S*normalCDF(-d1);
                    }
                    case OptionKind::Combo: {
                        if (Ks.size() == weights.size() && Ks.size() == isCall.size()) {
                            double sum = 0.0;
                            for (int i = 0; i < Ks.size(); ++i) {
                                const double d1 = (std::log(S/Ks[i]) + (r + sig*sig/2)*t)/(sig*std::sqrt(t));
                                const double d2 = d1 - sig*std::sqrt(t);
                                if (isCall[i]) {
                                    sum += weights[i]*(S*normalCDF(d1) - Ks[i]*std::exp(-r*t)*normalCDF(d2));
                                } else {
                                    sum += weights[i]*(Ks[i]*std::exp(-r*t)*normalCDF(-d2) - S*normalCDF(-d1));
                                };
                            };
                            return sum;
                        } else {
                            throw runtime_error("Class definitions not valid");
                        };
                    }
                    case OptionKind::Exotic: {
                        throw runtime_error("Black-Scholes doesn't work for this option type.");
                        break;
                    }
                };
            } else {
                throw runtime_error("Time to maturity can't be negative.");
            };
        };
        double delta(const Option& option, const MarketData& marketData) const override {
            double S = marketData.spot_price();
            double sig = marketData.volatility();
            double r = marketData.risk_free_rate();
            double t = marketData.maturity();
            OptionKind kind = option.kind();
            vector<double> Ks = option.strikes();
            vector<double> weights = option.weights();
            vector<bool> isCall = option.isCall();

            if (t == 0) {
                switch (kind) {
                    case OptionKind::VanillaCall: {
                        return (S>Ks[0]?1.0:0.0);
                    }
                    case OptionKind::VanillaPut: {
                        return (S<Ks[0]?-1.0:0.0);
                    }
                    case OptionKind::Combo: {
                        if (Ks.size() == weights.size() && Ks.size() == isCall.size()) {
                            double sum = 0.0;
                            for (int i = 0; i < Ks.size(); ++i) {
                                if (isCall[i]) {
                                    sum += (S>Ks[i]?1.0:0.0);
                                } else {
                                    sum += (S<Ks[i]?-1.0:0.0);
                                };
                            };
                            return sum;
                        } else {
                            throw runtime_error("Class definitions not valid");
                        };
                    }
                    case OptionKind::Exotic: {
                        throw runtime_error("Black-Scholes doesn't work for this option type.");
                        break;
                    }
                };
            } else if (t > 0) {
                switch (kind) {
                    case OptionKind::VanillaCall: {
                        const double d1 = (std::log(S/Ks[0]) + (r + sig*sig/2)*t)/(sig*std::sqrt(t));
                        return normalCDF(d1);
                    }
                    case OptionKind::VanillaPut: {
                        const double d1 = (std::log(S/Ks[0]) + (r + sig*sig/2)*t)/(sig*std::sqrt(t));
                        return normalCDF(d1) - 1;
                    }
                    case OptionKind::Combo: {
                        if (Ks.size() == weights.size() && Ks.size() == isCall.size()) {
                            double sum = 0.0;
                            for (int i = 0; i < Ks.size(); ++i) {
                                const double d1 = (std::log(S/Ks[i]) + (r + sig*sig/2)*t)/(sig*std::sqrt(t));
                                if (isCall[i]) {
                                    sum += weights[i]*normalCDF(d1);
                                } else {
                                    sum += weights[i]*(normalCDF(d1) - 1);
                                };
                            };
                            return sum;
                        } else {
                            throw runtime_error("Class definitions not valid");
                        };
                    }
                    case OptionKind::Exotic: {
                        throw runtime_error("Black-Scholes doesn't work for this option type.");
                        break;
                    }
                };
            } else {
                throw runtime_error("Time to maturity can't be negative.");
            };
        };
        double gamma(const Option& option, const MarketData& marketData) const override {
            double S = marketData.spot_price();
            double sig = marketData.volatility();
            double r = marketData.risk_free_rate();
            double t = marketData.maturity();
            OptionKind kind = option.kind();
            vector<double> Ks = option.strikes();
            vector<double> weights = option.weights();
            vector<bool> isCall = option.isCall();

            if (t==0) {
                return 0.0;
            } else if (t > 0) {
                switch (kind) {
                    case OptionKind::VanillaCall: {
                        const double d1 = (std::log(S/Ks[0]) + (r + sig*sig/2)*t)/(sig*std::sqrt(t));
                        return normalPDF(d1)/(S*sig*std::sqrt(t));
                    }
                    case OptionKind::VanillaPut: {
                        const double d1 = (std::log(S/Ks[0]) + (r + sig*sig/2)*t)/(sig*std::sqrt(t));
                        return normalPDF(d1)/(S*sig*std::sqrt(t));
                    }
                    case OptionKind::Combo: {
                        if (Ks.size() == weights.size() && Ks.size() == isCall.size()) {
                            double sum = 0.0;
                            for (int i = 0; i < Ks.size(); ++i) {
                                const double d1 = (std::log(S/Ks[i]) + (r + sig*sig/2)*t)/(sig*std::sqrt(t));
                                sum += weights[i]*(normalPDF(d1)/(S*sig*std::sqrt(t)));
                            };
                            return sum;
                        } else {
                            throw runtime_error("Class definitions not valid");
                        };
                    }
                    case OptionKind::Exotic: {
                        throw runtime_error("Black-Scholes doesn't work for this option type.");
                        break;
                    }
                };
            } else {
                throw runtime_error("Time to maturity can't be negative.");
            };
        };
        double rho(const Option& option, const MarketData& marketData) const override {
            double S = marketData.spot_price();
            double sig = marketData.volatility();
            double r = marketData.risk_free_rate();
            double t = marketData.maturity();
            OptionKind kind = option.kind();
            vector<double> Ks = option.strikes();
            vector<double> weights = option.weights();
            vector<bool> isCall = option.isCall();

            if (t==0) {
                return 0.0;
            } else if (t > 0) {
                switch (kind) {
                    case OptionKind::VanillaCall: {
                        const double d2 = (std::log(S/Ks[0]) + (r - sig*sig/2)*t)/(sig*std::sqrt(t));
                        return Ks[0]*t*std::exp(-r*t)*normalCDF(d2);
                    }
                    case OptionKind::VanillaPut: {
                        const double d2 = (std::log(S/Ks[0]) + (r - sig*sig/2)*t)/(sig*std::sqrt(t));
                        return -Ks[0]*t*std::exp(-r*t)*normalCDF(-d2);
                    }
                    case OptionKind::Combo: {
                        if (Ks.size() == weights.size() && Ks.size() == isCall.size()) {
                            double sum = 0.0;
                            for (int i = 0; i < Ks.size(); ++i) {
                                const double d2 = (std::log(S/Ks[i]) + (r - sig*sig/2)*t)/(sig*std::sqrt(t));
                                if (isCall[i]) {
                                    sum += weights[i]*Ks[i]*t*std::exp(-r*t)*normalCDF(d2);
                                } else {
                                    sum += -weights[i]*Ks[i]*t*std::exp(-r*t)*normalCDF(-d2);
                                };
                            };
                            return sum;
                        } else {
                            throw runtime_error("Class definitions not valid");
                        };
                    }
                    case OptionKind::Exotic: {
                        throw runtime_error("Black-Scholes doesn't work for this option type.");
                        break;
                    }
                };
            } else {
                throw runtime_error("Time to maturity can't be negative.");
            };
            
        };
        double vega(const Option& option, const MarketData& marketData) const override {
            double S = marketData.spot_price();
            double sig = marketData.volatility();
            double r = marketData.risk_free_rate();
            double t = marketData.maturity();
            OptionKind kind = option.kind();
            vector<double> Ks = option.strikes();
            vector<double> weights = option.weights();
            vector<bool> isCall = option.isCall();

            if (t==0) {
                return 0.0;
            } else if (t > 0) {
                switch (kind) {
                    case OptionKind::VanillaCall: {
                        const double d1 = (std::log(S/Ks[0]) + (r + sig*sig/2)*t)/(sig*std::sqrt(t));
                        return S*std::sqrt(t)*normalPDF(d1);
                    }
                    case OptionKind::VanillaPut: {
                        const double d1 = (std::log(S/Ks[0]) + (r + sig*sig/2)*t)/(sig*std::sqrt(t));
                        return S*std::sqrt(t)*normalPDF(d1);
                    }
                    case OptionKind::Combo: {
                        if (Ks.size() == weights.size() && Ks.size() == isCall.size()) {
                            double sum = 0.0;
                            for (int i = 0; i < Ks.size(); ++i) {
                                const double d1 = (std::log(S/Ks[i]) + (r + sig*sig/2)*t)/(sig*std::sqrt(t));
                                sum += weights[i]*S*std::sqrt(t)*normalPDF(d1);
                            };
                            return sum;
                        } else {
                            throw runtime_error("Class definitions not valid");
                        };
                    }
                    case OptionKind::Exotic: {
                        throw runtime_error("Black-Scholes doesn't work for this option type.");
                        break;
                    }
                };
            } else {
                throw runtime_error("Time to maturity can't be negative.");
            };
        };
};

class MC_Pricer : public OptionPricer {
    public:
        explicit MC_Pricer(size_t nSim=100000, double relBump=1e-3, double absBump=1e-4) : nSim_(nSim), relBump_(relBump), absBump_(absBump) {};
        double price(const Option& option, const MarketData& marketData) const override {
            auto Zs = generateZs_();
            return priceFromZs_(option,marketData,Zs);
        };
        double delta(const Option& option, const MarketData& marketData) const override {
            double S = marketData.spot_price();
            auto Zs = generateZs_();
            double h = std::max(1e-8,relBump_*S);

            MarketData up (S+h,marketData.volatility(),marketData.risk_free_rate(),marketData.maturity());
            MarketData down (S-h,marketData.volatility(),marketData.risk_free_rate(),marketData.maturity());

            return (priceFromZs_(option,up,Zs) - priceFromZs_(option,down,Zs))/(2.0*h);

        };
        double gamma(const Option& option, const MarketData& marketData) const override {
            double S = marketData.spot_price();
            auto Zs = generateZs_();
            double h = std::max(1e-8,relBump_*S);

            MarketData up (S+h,marketData.volatility(),marketData.risk_free_rate(),marketData.maturity());
            MarketData down (S-h,marketData.volatility(),marketData.risk_free_rate(),marketData.maturity());

            return (priceFromZs_(option,up,Zs) - 2*priceFromZs_(option,marketData,Zs) + priceFromZs_(option,down,Zs))/(h*h);
        };
        double rho(const Option& option, const MarketData& marketData) const override {
            double r = marketData.risk_free_rate();
            auto Zs = generateZs_();
            double h = absBump_;

            MarketData up (marketData.spot_price(),marketData.volatility(),r+h,marketData.maturity());
            MarketData down (marketData.spot_price(),marketData.volatility(),r-h,marketData.maturity());

            return (priceFromZs_(option,up,Zs) - priceFromZs_(option,down,Zs))/(2.0*h);
        };
        double vega(const Option& option, const MarketData& marketData) const override {
            double sig = marketData.volatility();
            auto Zs = generateZs_();
            double h = absBump_;

            MarketData up (marketData.spot_price(),sig+h,marketData.risk_free_rate(),marketData.maturity());
            MarketData down (marketData.spot_price(),sig-h,marketData.risk_free_rate(),marketData.maturity());

            return (priceFromZs_(option,up,Zs) - priceFromZs_(option,down,Zs))/(2.0*h);
        };
    private:
        size_t nSim_;
        double relBump_;
        double absBump_;

        std::vector<double> generateZs_() const {

            std::mt19937 gen(std::random_device{}());
            std::normal_distribution<double> dist(0.0,1.0);

            std::vector<double> Zs;
            Zs.reserve(nSim_);

            for (size_t i =0; i < nSim_; ++i) {
                Zs.push_back(dist(gen));
            };

            return Zs;
        };
        double priceFromZs_(const Option& option, const MarketData& marketData, const std::vector<double>& Zs) const {

            double S = marketData.spot_price();
            double sig = marketData.volatility();
            double r = marketData.risk_free_rate();
            double t = marketData.maturity();

            double sum = 0.0;

            for (double Z : Zs) {
                double S_t = S*std::exp((r - 0.5*sig*sig)*t + sig*std::sqrt(t)*Z);
                sum += option.payoff(S_t);
            }

            return (std::exp(-r*t)*sum)/nSim_;
        };
        
};

double normalCDF(double x) {
    return (1 + std::erf(x/std::sqrt(2)))/2;
};

double normalPDF(double x) {
    return std::exp(-0.5*x*x)/std::sqrt(2.0*M_PI);
};

int main() {

    MarketData market(100,0.20,0.05,1.0); // current stock price: 100, volatility: 20%, risk free rate: 5%, maturity: 1 year

    EuropeanCall call(120);
    EuropeanPut put(120);
    Straddle straddle(120);
    Butterfly butterfly(90,120,140);

    BS_Pricer bs_pricer;
    MC_Pricer mc_pricer;

    std::cout << "European call option price and Greeks (delta,gamma,rho,vega): " << std::endl;
    std::cout << bs_pricer.price(call,market) << " with " << "(" << bs_pricer.delta(call,market) << "," << bs_pricer.gamma(call,market) << "," << bs_pricer.rho(call,market) << "," << bs_pricer.vega(call,market) << ")" << " - Black-Scholes pricer" << std::endl;
    std::cout << mc_pricer.price(call,market) << " with " << "(" << mc_pricer.delta(call,market) << "," << mc_pricer.gamma(call,market) << "," << mc_pricer.rho(call,market) << "," << mc_pricer.vega(call,market) << ")" << " - Monte-Carlo pricer" << std::endl;
    std::cout << "\n";
    std::cout << "European put option price and Greeks (delta,gamma,rho,vega): " << std::endl;
    std::cout << bs_pricer.price(put,market) << " with " << "(" << bs_pricer.delta(put,market) << "," << bs_pricer.gamma(put,market) << "," << bs_pricer.rho(put,market) << "," << bs_pricer.vega(put,market) << ")" << " - Black-Scholes pricer" << std::endl;
    std::cout << mc_pricer.price(put,market) << " with " << "(" << mc_pricer.delta(put,market) << "," << mc_pricer.gamma(put,market) << "," << mc_pricer.rho(put,market) << "," << mc_pricer.vega(put,market) << ")" << " - Monte-Carlo pricer" << std::endl;
    std::cout << "\n";
    std::cout << "Straddle option price and Greeks (delta,gamma,rho,vega): " << std::endl;
    std::cout << bs_pricer.price(straddle,market) << " with " << "(" << bs_pricer.delta(straddle,market) << "," << bs_pricer.gamma(straddle,market) << "," << bs_pricer.rho(straddle,market) << "," << bs_pricer.vega(straddle,market) << ")" << " - Black-Scholes pricer" << std::endl;
    std::cout << mc_pricer.price(straddle,market) << " with " << "(" << mc_pricer.delta(straddle,market) << "," << mc_pricer.gamma(straddle,market) << "," << mc_pricer.rho(straddle,market) << "," << mc_pricer.vega(straddle,market) << ")" << " - Monte-Carlo pricer" << std::endl;
    std::cout << "\n";
    std::cout << "Butterfly option price and Greeks (delta,gamma,rho,vega): " << std::endl;
    std::cout << bs_pricer.price(butterfly,market) << " with " << "(" << bs_pricer.delta(butterfly,market) << "," << bs_pricer.gamma(butterfly,market) << "," << bs_pricer.rho(butterfly,market) << "," << bs_pricer.vega(butterfly,market) << ")" << " - Black-Scholes pricer" << std::endl;
    std::cout << mc_pricer.price(butterfly,market) << " with " << "(" << mc_pricer.delta(butterfly,market) << "," << mc_pricer.gamma(butterfly,market) << "," << mc_pricer.rho(butterfly,market) << "," << mc_pricer.vega(butterfly,market) << ")" << " - Monte-Carlo pricer" << std::endl;
    std::cout << "\n";

}