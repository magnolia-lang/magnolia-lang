package tests.inputs.modeCheck;

implementation modeCheck_I = {
    type T;

    // TODO: overloading on mode should not work. Uncomment when error is
    //       correctly thrown.
    // procedure foo(out in: T);
    // procedure foo(obs in: T);

    procedure in_obs(obs in: T);
    procedure in_out(out in: T);
    procedure in_upd(upd in: T);

    // obs input
    procedure call_obs_obs(obs in: T) = call in_obs(in); // should succeed
    procedure call_obs_out(obs in: T) = call in_out(in); // should fail
    procedure call_obs_upd(obs in: T) = call in_upd(in); // should fail

    // out input
    procedure call_out_obs(out in: T) = call in_obs(in); // should fail
    procedure call_out_out(out in: T) = call in_out(in); // should succeed
    procedure call_out_upd(out in: T) = call in_upd(in); // should fail

    // upd input
    procedure call_upd_obs(upd in: T) = call in_obs(in); // should succeed
    procedure call_upd_out(upd in: T) = call in_out(in); // should succeed
    procedure call_upd_upd(upd in: T) = call in_upd(in); // should succeed

    // test mismatch across several arguments
    procedure in_2_obs(obs in1: T, obs in2: T);
    procedure call_out_2_obs(out in: T) = call in_2_obs(in, in);
}
