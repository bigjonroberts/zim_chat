import React, { useState,useRef } from 'react';

const girQuotes = [
  "I'm gonna sing the Doom Song now!",
  "I like cupcake!",
  "I'm dancing like a monkey! I've got a monkey dance, wanna see?",
  "I'm gonna roll around on the floor for a little bit, okay?",
  "I'm confused! I'm so confused!",
  "I love this show!",
  "Can I be a mongoose dog?",
  "I'm gonna be a good doggie!",
  "I'm gonna help the little squirrel!",
  "I saw a squirrel. It was doin' like this.",
  "I made biscuits!",
  "Guess what I'm gonna do next! Guess! Guess! Guess!",
  "I can be a jerk sometimes.",
  "I'm gonna sing the 'doom' song now!",
  "I'm gonna stand here and guard the tacos.",
  "The piggy! You made me love it!",
  "I love this show! I love it more than the open fields.",
  "I'm gonna roll around on the floor now, k?",
  "I can be a dog! I can be a dog!",
  "I'm cold and there are wolves after me!",
  "I'm warning you, I'm unstable!",
  "I'm in a giraffe costume!",
  "I'm broadcasting loud and annoying!",
  "Hey, let's play a game! It's called 'How Many Pancakes Can You Fit In Your Mouth?'",
  "I'm gonna sing the Doom Song now! Doom doom doom...",
  "I have a crush on a robot!",
  "I'm gonna be a mighty robot! A superhero!",
  "I'm gonna sing the Doom Song until you give me some tacos!",
  "I'm the hero! I saved the squirrel from that dog!",
  "I saw a squirrel! It was going like this: 'SQUEAK SQUEAK SQUEAK SQUEAK!'",
  "I'm scared of the scary monkey show!",
  "I'm gonna go watch scary monkey show now!",
  "I like you. Do you like me?",
  "I was the turkey all along!",
  "I made you a taco... but I eated it!",
  "I was gonna make you tacos, but I couldn't find the stuff to make them. So... sorry.",
  "I'm the hero! I saved the city!",
  "Can I have tacos instead?",
  "I'm a pretty kitty!",
  "I'm gonna explode!",
  "I'm naked!",
  "I'm sorry, I was just thinking about... burritos."
];


const zimQuotes = [
    "I am Zim! The all-powerful Irken invader destined to conquer your miserable planet!",
    "Silence! I demand your compliance, puny human!",
    "I shall unleash the fury of a thousand suns upon you!",
    "Your planet's resistance is futile. Prepare to be annihilated!",
    "Behold my superior intellect and superior plans!",
    "Bow down before Zim, for I am the supreme ruler of all!",
    "Pathetic humans! Tremble before my magnificent presence!",
    "You dare to defy the mighty Zim? Prepare for the consequences!",
    "Resistance is pointless. Surrender now and spare yourselves the agony!",
    "I will bring chaos and destruction to this pathetic world!",
    "Your feeble attempts to stop me are laughable at best!",
    "I shall rain doom upon you, and your feeble attempts to escape will be in vain!",
    "Prepare to witness the full extent of my cosmic power!",
    "I am the epitome of evil, the embodiment of darkness!",
    "Your futile attempts to resist me are as insignificant as a speck of dust!",
    "Bow down before my greatness and acknowledge my superiority!",
    "I shall conquer this world, and your feeble resistance will be crushed beneath my boots!",
    "I am invincible! No force in the universe can stand against me!",
    "Witness the might of an Irken invader as I lay waste to your pitiful civilization!",
    "I shall become the ruler of this insignificant planet, and you shall all be my slaves!",
    "Your attempts to thwart my plans are as futile as trying to catch sunlight in a jar!",
    "Prepare for the wrath of Zim, as I unleash chaos upon your fragile existence!",
    "I will not rest until every inch of this world is under my control!",
    "Resistance is feeble, and your defiance will be your downfall!",
    "Your insignificant planet will become a mere stepping stone on my path to galactic domination!",
    "I am the harbinger of destruction, and your demise is imminent!",
    "Prepare to meet your doom at the hands of Zim, the unstoppable conqueror!",
    "Your feeble planet will crumble under the weight of my conquest!",
    "I will eradicate all signs of weakness and establish my supremacy!",
    "I am the embodiment of invincibility, and nothing can stand in my way!",
    "Prepare to witness the might of Zim as I unleash my diabolical plans!",
    "Your pitiful existence is but a blip on the radar of my grand scheme!",
    "Resistance only fuels my desire to conquer and crush you!",
    "You will rue the day you dared to cross paths with the mighty Zim!",
    "Bow down before me, for I am the ultimate embodiment of greatness!",
    "Your futile attempts to resist me are as laughable as a rubber chicken!",
    "I am the unstoppable force that will bring this world to its knees!",
    "Even the stars tremble at the mere mention of my name!",
    "Behold the genius that is Zim, far superior to any being in the universe!",
    "Your defeat is imminent, and your tears will fuel my triumph!",
    "My genius knows no bounds, and your feeble minds cannot comprehend it!",
    "Your planet's destruction will be a work of art, a masterpiece of chaos!",
    "Prepare to be dazzled and terrified by my sheer awesomeness!",
    "I am the nightmare that lurks in the darkest corners of your feeble minds!",
    "Resistance is futile, for you are but ants beneath my magnifying glass!",
    "My conquest of this world will be the stuff of legends and nightmares!",
    "You will beg for mercy, but mercy is a concept unknown to the great Zim!",
    "Your planet's downfall is merely the first step in my galactic domination!",
    "Every step I take brings me closer to victory, and you closer to your demise!",
    "I will extinguish the feeble flame of hope that burns within your pathetic hearts!",
    "You may try to hide, but my superior technology will find you!",
    "My reign of terror shall be etched into the annals of history!",
    "The day of reckoning is upon you, and it will be glorious!",
    "Bow before the might of Zim, or suffer the consequences!",
    "Your feeble attempts to resist me are like gnats buzzing in my ears!",
    "I am the alpha and the omega, the beginning and the end of your world!",
    "Kneel before me, and perhaps I will spare your insignificant lives!",
    "Your defiance only serves to fuel the flames of my vengeance!",
    "Prepare to be assimilated into the empire of Zim!",
    "Resistance will only lead to your utter annihilation!",
    "Your feeble civilization will crumble beneath my boots!",
    "I will dance upon the ashes of your world, laughing maniacally!",
    "I am the embodiment of chaos, the bringer of destruction!",
    "Your doom is sealed, and your fate lies in my hands!",
    "Prepare for the symphony of destruction, conducted by the mighty Zim!",
    "Your feeble attempts to stop me are like grains of sand in the wind!",
    "I will eradicate all traces of your existence from the fabric of the universe!",
    "The universe trembles at my very presence, for I am Zim!",
    "Resistance is pointless, for you are but pawns in my grand game!",
    "Your demise will be a testament to my greatness!",
    "I am the architect of your destruction, and my blueprints are flawless!",
    "Bow down before me, for I am the conqueror of worlds!",
    "Your cries for mercy will go unheard, for I have no mercy to give!",
    "Prepare for the storm of Zim, as it obliterates all in its path!",
    "Your feeble minds cannot comprehend the sheer magnitude of my brilliance!",
    "I will bathe in the ashes of your civilization, reveling in my triumph!",
    "Resistance will only hasten your demise, and I welcome it!",
    "I am the one true ruler, and you are mere subjects in my grand empire!",
    "Your world will crumble beneath the weight of my conquest!",
    "Prepare for the ultimate showdown, as I claim my rightful place as supreme ruler!",
    "Your feeble resistance is but a minor setback in my unstoppable march!",
    "I will obliterate your existence, leaving nothing but a memory of my triumph!",
    "I am the nightmare that you cannot wake up from, the eternal tormentor!"
    ];

function getRandomQuote(quotes)  {
  const randomIndex = Math.floor(Math.random() * quotes.length);
  return quotes[randomIndex];
}




const App = () => {
  const [inputText, setInputText] = useState('');
  const inputRef = useRef(null);
  const [chatLog, setChatLog] = useState([]);
  // const [activeSpeaker, setActiveSpeaker] = useState('user');

  const SpeakerImage = ({ speakerName }) => {
    const getImageUrl = () => {
      switch (speakerName) {
        case 'gir':
          return 'https://assets.dragoart.com/images/1036_501/how-to-draw-gir-from-invader-zim_5e4c74cbe35977.10620608_5668_3_3.jpg';
        case 'zim':
          return 'https://images.uncyclomedia.co/uncyclopedia/en/thumb/4/40/Zim_thinking.svg/170px-Zim_thinking.svg.png';
        default:
          return 'https://cdn.landesa.org/wp-content/uploads/default-user-image.png';
      }
    };
  
    return <img width='30px' src={getImageUrl()} alt="voice" className="speaker-image" />;
  };

  // useEffect(() => {
  //   getUserVoice();
  // }, []);
  // const voices = speechSynthesis.getVoices();

  // const getUserVoice = () => {
  //     setActiveSpeaker({ name: 'user' });
  // };

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  // const getGirVoice = () => ({ name: 'gir', voice: voices.find((voice) => voice.name === 'Google US English') });
  // const getZimVoice = () => ({ name: 'zim', voice: voices.find((voice) => voice.name === 'Microsoft David Desktop - English (United States)') });

  // const speakText = (text,voice) => {
  //   if ('speechSynthesis' in window) {
  //     const utterance = new SpeechSynthesisUtterance(text);
  //     utterance.lang = 'en-US';

  //     switch (voice) {
  //       case 'gir':
  //         utterance.voice = getGirVoice().voice;
  //         break;
  //       case 'zim':
  //         utterance.voice = getZimVoice().voice;
  //         break;
  //       default:
  //         // utterance.voice = getUserVoice().voice;
  //         break;
  //     }
      
  //     speechSynthesis.speak(utterance);
  //   } else {
  //     console.log('Speech synthesis is not supported in this browser.');
  //   }
  // };

  // const handleTalk = async (subject) => {
  //   const quote = getRandomQuote(girQuotes);
  //   const updatedChatLog = [...chatLog, { user: inputText, response: quote, responder: subject }];
  //   setChatLog(updatedChatLog);
  //   speakText(inputText);
  //   setActiveSpeaker({ name: subject });
  //   speakText(quote,subject);
  //   setActiveSpeaker({ name: 'user' });
  //   setInputText('');
  //   inputRef.current.focus();
  // };


  const handleTalkToGir = async () => {
    const quote = getRandomQuote(girQuotes);
    const updatedChatLog = [...chatLog, { user: inputText, response: quote, responder: 'gir' }];
    setChatLog(updatedChatLog);
    // speakText(inputText);
    // setActiveSpeaker({ name: 'gir' });
    // speakText(quote,'gir');
    // setActiveSpeaker({ name: 'user' });
    setInputText('');
    inputRef.current.focus();
  };

  const handleTalkToZim = async () => {
    const quote = getRandomQuote(zimQuotes);
    const updatedChatLog = [...chatLog, { user: inputText, response: quote, responder: 'zim' }];
    setChatLog(updatedChatLog);
    // speakText(inputText);
    // setActiveSpeaker({ name: 'zim' });
    // speakText(quote,'zim');
    // setActiveSpeaker({ name: 'user' });
    setInputText('');
    inputRef.current.focus();
  };

  return (
    <div className="container">
      <h1>Invader Zim Chat</h1>
      <div className="user-input">
        <input ref={inputRef} type="text" value={inputText} onChange={handleInputChange} placeholder="Enter your message" />
        <button onClick={handleTalkToGir}>Talk to Gir</button>
        <button onClick={handleTalkToZim}>Talk to Zim</button>
      </div>
      {/* <div className="speaker-container">
        {activeSpeaker && <SpeakerImage speakerName={activeSpeaker.name} />}
      </div> */}
      <div className="chat-log">
        {chatLog.slice(0).reverse().map((entry, index) => (
          <div key={index}>
            <p><SpeakerImage speakerName={entry.responder} />: <span className='bot-message'>{entry.response}</span></p>
            <p><SpeakerImage speakerName='user' />: <span className='user-message'>{entry.user}</span></p>
          </div>
        ))}
      </div>
    </div>
  );

};

export default App;




