import unittest
from local_trpg.modules import load_module
from local_trpg.engine.rules import initial_state,Engine
from local_trpg.engine.reducer import apply_event
from local_trpg.schemas import ActionIntent

class Adventure(unittest.TestCase):
 def setUp(self):self.m=load_module();self.start=initial_state(self.m,'试玩');self.e=Engine(self.start,self.m,'route',lambda k:[1,2])
 def go(self,place):self.e.move('player',place)
 def do(self,action,actor='player',targets=()):
  self.e.advance(1);self.e.module_action(action,actor,list(targets));self.e.finish_clock()
 def inspect(self,refs,place):
  self.go(place);self.e.resolve_player(ActionIntent(actor_id='player',kind='INSPECT',evidence_refs=refs));self.e.finish_clock()
 def test_route_a_all_saved(self):
  self.inspect(['E01','E04'],'engine')
  self.e.advance(1);self.e.give_evidence('E03',['player'],'mechanic');self.e.finish_clock()
  self.inspect(['E02'],'bridge')
  self.e.advance(1);self.e.give_evidence('E06',['player'],'comms');self.e.finish_clock()
  self.e.advance(1);self.e.give_evidence('E05',['player'],'doctor');self.e.finish_clock()
  self.go('lifeboat');self.do('take_cold_cases')
  self.go('cargo');self.e.give_evidence('E07',['doctor']);self.do('transfer_cells','doctor')
  self.do('isolate_branch','mechanic');self.go('engine');self.do('restore_main_loop')
  self.assertEqual(self.e.state.tick,9);self.assertEqual(self.e.state.cells[0]['expiry'],25)
  self.e.module_action('choose_station','player',[])
  o=self.e.state.outcome;self.assertEqual((o['crew_saved'],o['cells_saved'],o['ship_saved'],o['arrival_tick']),(6,4,True,14))
  replay=self.start
  for ev in self.e.events:replay=apply_event(replay,ev)
  self.assertEqual(replay,self.e.state)
 def test_route_b_bad_rolls_do_not_softlock(self):
  self.inspect(['E01','E04'],'engine')
  self.e.resolve_player(ActionIntent(actor_id='player',kind='DECEIVE',target_ids=['mechanic'],spoken_text='我亲眼看到机械师关闭保护',channel='broadcast'));self.e.finish_clock()
  self.do('isolate_branch');self.assertFalse(self.e.state.flags['cargo_branch_isolated']);self.assertEqual(self.e.state.relations['mechanic'],-2)
  self.do('begin_safe_isolation');self.do('complete_safe_isolation');self.do('restore_main_loop')
  self.e.module_action('choose_station','player',[]);o=self.e.state.outcome
  self.assertEqual((o['crew_saved'],o['cells_saved'],o['ship_saved'],o['arrival_tick']),(6,0,True,11))
 def test_route_c_actual_boarding_and_ship_deadline(self):
  self.do('send_distress','comms');self.e.advance(1);self.e.give_evidence('E06',['player'],'comms');self.e.finish_clock()
  self.go('lifeboat');self.do('take_cold_cases')
  self.e.advance(1);self.e.give_evidence('E07',['doctor']);self.e.give_evidence('E05',['player'],'doctor');self.e.finish_clock()
  self.go('cargo');self.do('transfer_cells','doctor');self.assertEqual(self.e.state.cells[0]['expiry'],23)
  self.e.give_evidence('E06',list(self.e.state.actors));self.do('prepare_pod','guard')
  self.do('assemble_crew');self.do('board_pod');self.do('launch_pod')
  o=self.e.state.outcome;self.assertEqual((o['crew_saved'],o['cells_saved'],o['ship_saved']),(6,4,False));self.assertTrue(self.e.state.flags['engine_destroyed'])
  self.assertEqual(len(self.e.state.fired_triggers),len(set(self.e.state.fired_triggers)))
  # Passing the abandoned ship's deadline must not first announce a second
  # rescue at tick15 and then overwrite it with the real arrival time.
  self.assertEqual(sum(event.kind=='ScenarioResolved' for event in self.e.events),1)
 def test_guessing_right_needs_no_evidence_gate(self):
  self.go('cargo');self.do('begin_safe_isolation');self.do('complete_safe_isolation');self.go('engine');self.do('restore_main_loop')
  self.assertTrue(self.e.state.flags['main_loop_restored'])
