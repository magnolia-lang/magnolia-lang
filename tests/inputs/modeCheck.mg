package tests.inputs.modeCheck;


concept ModeCheck = {

    type T;

    //TODO this should not work, overloading on mode 
    procedure foo (upd a:T);
    //procedure foo (obs a:T);

};

implementation Check = {
    use ModeCheck;
    type X;
    procedure in_obs (obs a:T);

    procedure in_upd (upd a:T);

    procedure in_out (out a:T);

    procedure call_obs_obs(obs n:T) = call in_obs(n);
    procedure call_obs_upd(obs n:T) = call in_upd(n); 
    procedure call_obs_out(obs n:T) = call in_out(n);


    procedure call_upd_obs(upd n:T) = call in_obs(n);
    procedure call_upd_upd(upd n:T) = call in_upd(n);
    procedure call_upd_out(upd n:T) = call in_out(n);


    procedure call_out_obs(out n:T) = call in_obs(n);
    procedure call_out_upd(out n:T) = call in_upd(n);
    procedure call_out_out(out n:T) = call in_out(n);

}



